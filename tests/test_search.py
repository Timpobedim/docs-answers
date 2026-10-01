from __future__ import annotations

from collections.abc import Sequence

import httpx
import pytest

from answers.documents import Section
from answers.index import Index, build
from answers.search import OllamaEmbeddings, SearchError, cosine, find


class FakeEmbeddings:
    """Вектор — счётчик ключевых слов: смысловая близость без нейросети, но проверяемо."""

    name = "фейковые"
    WORDS = ("отпуск", "командировк", "пропуск", "закупк", "пароль", "отгул")

    def __init__(self) -> None:
        self.asked: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.asked.append(list(texts))
        return [[float(text.lower().count(word)) for word in self.WORDS] for text in texts]


async def test_without_embeddings_it_is_plain_bm25(index: Index) -> None:
    hits = await find(index, "суточные в командировке", limit=3)
    assert "Суточные и проживание" in hits[0].section.reference
    assert await find(index, "рецепт борща") == []


async def test_embeddings_add_meaning() -> None:
    sections = [
        Section("отпуска.md", ("Отпуска", "Сколько дней"), "Ежегодный оплачиваемый отпуск — 28 календарных дней.", 1),
        Section("закупки.md", ("Закупки", "Лимиты"), "Закупки до 50 000 рублей согласует руководитель.", 1),
    ]
    embeddings = FakeEmbeddings()
    vectors = await embeddings.embed([f"{section.title}. {section.text}" for section in sections])
    index = build(sections, vectors)
    hits = await find(index, "отгул и отпуск", embeddings=embeddings, limit=2, weight=0.9)
    assert "Сколько дней" in hits[0].section.reference
    assert embeddings.asked[-1] == ["отгул и отпуск"]


def test_cosine() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([], [1]) == 0.0


async def test_ollama_embeddings_request() -> None:
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        vectors = await OllamaEmbeddings("qwen3:4b", client=client).embed(["а", "б"])
    finally:
        await client.aclose()
    assert seen["url"] == "http://127.0.0.1:11434/api/embed"
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


async def test_embeddings_failures_are_explained() -> None:
    broken = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500, text="упал")))
    try:
        with pytest.raises(SearchError, match="500"):
            await OllamaEmbeddings("qwen3:4b", client=broken).embed(["а"])
    finally:
        await broken.aclose()

    short = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"embeddings": [[1.0]]}))
    )
    try:
        with pytest.raises(SearchError, match="не то количество"):
            await OllamaEmbeddings("qwen3:4b", client=short).embed(["а", "б"])
    finally:
        await short.aclose()
