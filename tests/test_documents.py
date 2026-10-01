from __future__ import annotations

from pathlib import Path

import pytest
from conftest import EXAMPLES

from answers.documents import MAX_PIECE, DocumentError, Section, load, read_text, split_sections

MARKDOWN = """# Отпуска

Общие слова про отпуска.

## Как согласовать
Заявление подаётся за 14 дней.

### Кто согласует
Руководитель, затем кадры.

## Отпуск за свой счёт
До 5 дней по семейным обстоятельствам.
"""
NUMBERED = """1. Общие положения
Регламент описывает порядок работы.

1.1 Область применения
Действует для всех подразделений.

2. Ответственность
Ответственный — руководитель подразделения.
"""


def test_headings_become_the_path() -> None:
    sections = split_sections("отпуска.md", MARKDOWN)
    assert [section.path for section in sections] == [
        ("Отпуска",),
        ("Отпуска", "Как согласовать"),
        ("Отпуска", "Как согласовать", "Кто согласует"),
        ("Отпуска", "Отпуск за свой счёт"),
    ]
    assert sections[1].reference == "отпуска.md · Отпуска → Как согласовать"
    assert sections[1].text == "Заявление подаётся за 14 дней."


def test_numbered_headings_also_work() -> None:
    sections = split_sections("регламент.txt", NUMBERED)
    assert [section.title for section in sections] == [
        "Общие положения",
        "Общие положения → Область применения",
        "Ответственность",
    ]


def test_long_section_is_split_by_sentences() -> None:
    sentence = "Это предложение про порядок согласования заявок и сроки их рассмотрения. "
    sections = split_sections("длинный.md", "# Раздел\n" + sentence * 40)
    assert len(sections) > 1
    assert [section.number for section in sections] == list(range(1, len(sections) + 1))
    assert all(len(section.text) <= MAX_PIECE + len(sentence) for section in sections)
    assert sections[1].reference.endswith("часть 2")


def test_docx_is_read_with_headings(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Пропуска", level=1)
    document.add_paragraph("Пропуск выдают на первом этаже.")
    document.add_heading("Гостевой пропуск", level=2)
    document.add_paragraph("Заказывает принимающий сотрудник.")
    path = tmp_path / "пропуска.docx"
    document.save(path)
    sections = split_sections(path.name, read_text(path))
    assert [section.title for section in sections] == ["Пропуска", "Пропуска → Гостевой пропуск"]


def test_corpus_of_examples(sections: list[Section]) -> None:
    documents = {section.document for section in sections}
    assert len(documents) == 5
    assert len(sections) >= 20
    assert all(section.text.strip() for section in sections)
    assert any("Отпуска → Сколько дней отпуска положено" in section.reference for section in sections)


def test_missing_and_empty_folders(tmp_path: Path) -> None:
    with pytest.raises(DocumentError, match="не найдена"):
        load(tmp_path / "нет-такой")
    (tmp_path / "пусто").mkdir()
    with pytest.raises(DocumentError, match="нет файлов"):
        load(tmp_path / "пусто")
    assert (EXAMPLES / "регламенты").exists()
