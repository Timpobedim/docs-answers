"""Командная строка: собрать индекс, спросить, замерить качество, показать демо, поднять сервис.

answers index --corpus регламенты --out индекс.json
answers ask "сколько дней отпуска"
answers quality --questions вопросы.json
answers demo
answers serve --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from collections.abc import Sequence
from pathlib import Path

from answers.answer import show
from answers.api import Service, create_app
from answers.config import Settings, SettingsError
from answers.documents import DocumentError, load
from answers.index import Index, build, save
from answers.index import load as load_index
from answers.llm import ModelError, NullModel
from answers.quality import QuestionsError, load_questions, measure, report
from answers.search import SearchError

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="answers", description="Поиск ответов по внутренним регламентам")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="собрать индекс по папке с регламентами")
    index.add_argument("--corpus", type=Path, help="папка с .md, .txt и .docx")
    index.add_argument("--out", type=Path, help="куда сохранить индекс")
    index.add_argument("--embeddings", action="store_true", help="посчитать эмбеддинги разделов")

    ask = commands.add_parser("ask", help="спросить по регламентам")
    ask.add_argument("question", help="вопрос")
    ask.add_argument("--index", type=Path, help="файл индекса")
    ask.add_argument("--limit", type=int, help="сколько разделов брать")

    quality = commands.add_parser("quality", help="замерить поиск на размеченных вопросах")
    quality.add_argument("--questions", type=Path, required=True, help="JSON со списком вопросов и разделов")
    quality.add_argument("--index", type=Path, help="файл индекса")

    commands.add_parser("demo", help="собрать индекс по примерам и ответить на несколько вопросов")

    serve = commands.add_parser("serve", help="поднять HTTP-сервис")
    serve.add_argument("--index", type=Path, help="файл индекса")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


async def _build(settings: Settings, corpus: Path, out: Path, with_embeddings: bool) -> Index:
    sections = load(corpus)
    vectors = None
    if with_embeddings:
        embeddings = settings.build_embeddings()
        if embeddings is None:
            raise SettingsError("Для --embeddings поставьте ANSWERS_EMBEDDINGS=ollama")
        vectors = await embeddings.embed([f"{section.title}. {section.text}" for section in sections])
    index = build(sections, vectors)
    save(index, out)
    return index


def _load(path: Path) -> Index:
    if not path.exists():
        raise DocumentError(f"Индекс {path} не найден — соберите его командой «answers index»")
    return load_index(path)


async def _ask(settings: Settings, index: Index, question: str, limit: int) -> str:
    service = Service(index, settings.build_model(), settings.build_embeddings(), limit, settings.weight)
    return show(await service.ask(question))


async def _demo(settings: Settings) -> int:
    corpus = EXAMPLES / "регламенты"
    index = build(load(corpus))
    print(f"Регламенты: {corpus}\nРазделов в индексе: {len(index.sections)}\n")
    service = Service(index, NullModel(), None, settings.limit, settings.weight)
    for question in (
        "сколько дней отпуска положено",
        "как заказать монитор на рабочее место",
        "что делать, если потерял пропуск",
        "можно ли пользоваться личной флешкой",
        "какая погода завтра",
    ):
        print(f"— {question}")
        print(show(await service.ask(question)).replace("\n", "\n  "), end="\n\n")
    questions = EXAMPLES / "вопросы.json"
    if questions.exists():
        print(report(await measure(index, load_questions(questions)), "Замер поиска на размеченных вопросах"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
    args = _parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.command == "index":
            corpus = args.corpus or settings.corpus
            out = args.out or settings.index_path
            index = asyncio.run(_build(settings, corpus, out, args.embeddings))
            print(f"Разделов: {len(index.sections)}. Индекс: {out}")
            return 0
        if args.command == "ask":
            index = _load(args.index or settings.index_path)
            print(asyncio.run(_ask(settings, index, args.question, args.limit or settings.limit)))
            return 0
        if args.command == "quality":
            index = _load(args.index or settings.index_path)
            questions = load_questions(args.questions)
            score = asyncio.run(measure(index, questions, embeddings=settings.build_embeddings(), limit=settings.limit))
            print(report(score))
            return 0
        if args.command == "demo":
            return asyncio.run(_demo(settings))
        import uvicorn  # локальный импорт: для остальных команд сервер не нужен

        service = Service(
            _load(args.index or settings.index_path),
            settings.build_model(),
            settings.build_embeddings(),
            settings.limit,
            settings.weight,
        )
        uvicorn.run(create_app(service), host=args.host, port=args.port, log_level="warning")
        return 0
    except (SettingsError, DocumentError, QuestionsError) as exc:
        print(exc, file=sys.stderr)
        return 2
    except (ModelError, SearchError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
