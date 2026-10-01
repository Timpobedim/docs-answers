"""Регламенты с диска → разделы, по которым можно искать и на которые можно сослаться.

Разбиение идёт по заголовкам, а не по 1000 символов подряд: человеку нужно показать, откуда взят
ответ, а «фрагмент №137» ничего не говорит. У каждого куска остаётся путь заголовков — «Отпуска →
Как согласовать», — и он же становится ссылкой в ответе.

Куски режутся по разделам, а не по длине, и знают свой заголовок, а не только имя файла.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUFFIXES = (".md", ".txt", ".docx")
MAX_PIECE = 900  # символов в куске: длинный раздел режется по предложениям
OVERLAP = 1  # сколько предложений повторить в начале следующего куска
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([А-ЯЁA-Z][^.]{3,80})$")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


class DocumentError(ValueError):
    pass


@dataclass(frozen=True)
class Section:
    document: str  # имя файла регламента
    path: tuple[str, ...]  # путь заголовков: («Отпуска», «Как согласовать»)
    text: str
    number: int  # номер куска внутри раздела, с 1

    @property
    def title(self) -> str:
        return " → ".join(self.path) if self.path else self.document

    @property
    def reference(self) -> str:
        tail = f", часть {self.number}" if self.number > 1 else ""
        return f"{self.document} · {self.title}{tail}"


def read_text(path: Path) -> str:
    """Текст регламента. .docx читается через python-docx, остальное — как обычный текст."""
    if path.suffix.lower() == ".docx":
        try:
            from docx import Document  # импорт здесь: без .docx зависимость не нужна
        except ImportError as exc:  # pragma: no cover — зависимость объявлена в pyproject
            raise DocumentError("Для .docx нужен пакет python-docx") from exc
        try:
            document = Document(str(path))
        except (OSError, ValueError, KeyError) as exc:
            raise DocumentError(f"Не удалось прочитать {path.name}: {exc}") from exc
        lines = []
        for paragraph in document.paragraphs:
            style = (paragraph.style.name if paragraph.style is not None else "") or ""
            text = paragraph.text.strip()
            if not text:
                continue
            level = next((int(char) for char in style if char.isdigit()), 0) if style.startswith("Heading") else 0
            lines.append(f"{'#' * level} {text}" if level else text)
        return "\n\n".join(lines)
    try:
        return path.read_text("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise DocumentError(f"Не удалось прочитать {path.name}: {exc}") from exc


def _pieces(text: str) -> list[str]:
    """Длинный раздел → куски по предложениям с небольшим нахлёстом."""
    body = " ".join(text.split())
    if len(body) <= MAX_PIECE:
        return [body] if body else []
    sentences = SENTENCE.split(body)
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        current.append(sentence)
        if sum(len(part) + 1 for part in current) >= MAX_PIECE:
            pieces.append(" ".join(current))
            current = current[-OVERLAP:] if OVERLAP else []
    rest = " ".join(current).strip()
    if rest and (not pieces or rest != pieces[-1]):
        pieces.append(rest)
    return pieces


def split_sections(document: str, text: str) -> list[Section]:
    """Текст регламента → разделы. Заголовки — «# Название» или «2.1 Название»."""
    path: list[str] = []
    body: list[str] = []
    sections: list[Section] = []

    def flush() -> None:
        for number, piece in enumerate(_pieces("\n".join(body)), start=1):
            sections.append(Section(document, tuple(path), piece, number))
        body.clear()

    for line in text.splitlines():
        heading = HEADING.match(line.strip())
        numbered = NUMBERED.match(line.strip()) if heading is None else None
        if heading is None and numbered is None:
            body.append(line)
            continue
        flush()
        if heading is not None:
            level, title = len(heading.group(1)), heading.group(2)
        else:
            assert numbered is not None  # noqa: S101 — ветка разобрана выше
            level, title = numbered.group(1).count(".") + 1, numbered.group(2)
        path = [*path[: level - 1], title.strip()]
    flush()
    return sections


def load(folder: Path) -> list[Section]:
    """Все регламенты из папки (включая вложенные) → разделы."""
    if not folder.exists():
        raise DocumentError(f"Папка с регламентами не найдена: {folder}")
    sections: list[Section] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            sections.extend(split_sections(path.name, read_text(path)))
    if not sections:
        raise DocumentError(f"В {folder} нет файлов {', '.join(SUFFIXES)} с текстом")
    return sections
