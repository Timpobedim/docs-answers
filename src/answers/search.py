"""Поиск: слова обязательно, смысл — если есть чем считать.

BM25 работает всегда и не требует ничего, кроме Python. Эмбеддинги подключаются отдельно: если рядом
поднята Ollama, к словам добавляется смысловая близость, и вопрос «сколько дней отпуска» находит
раздел, где написано «продолжительность ежегодного оплачиваемого отпуска».

Поиск не привязан к Chroma, HuggingFace-эмбеддингам и кросс-энкодеру: работает без них.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import httpx

from answers.index import Hit, Index, search

TIMEOUT = httpx.Timeout(120.0, connect=5.0)
MIN_SCORE = 1.0  # ниже этого BM25 не считаем находкой
WEIGHT = 0.5  # вклад смысловой близости в общий балл


class SearchError(RuntimeError):
    pass


class Embeddings(Protocol):
    name: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OllamaEmbeddings:
    """Локальные эмбеддинги: POST /api/embed. Модель по умолчанию — nomic-embed-text."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        own = self._client is None
        session = self._client or httpx.AsyncClient(timeout=TIMEOUT)
        try:
            response = await session.post(
                f"{self.base_url}/api/embed", json={"model": self.model, "input": list(texts)}
            )
            if response.status_code >= 400:
                raise SearchError(f"Эмбеддинги: сервер ответил {response.status_code}")
            data = response.json()
        except httpx.HTTPError as exc:
            raise SearchError(f"Эмбеддинги недоступны: {exc}") from exc
        finally:
            if own:
                await session.aclose()
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise SearchError("Эмбеддинги: сервер вернул не то количество векторов")
        return [[float(value) for value in vector] for vector in vectors]


def cosine(first: Sequence[float], second: Sequence[float]) -> float:
    if not first or not second or len(first) != len(second):
        return 0.0
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    length = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))
    return dot / length if length else 0.0


def _normalized(scores: dict[int, float]) -> dict[int, float]:
    """Баллы разной природы сравнивать нельзя — приводим каждый список к 0…1."""
    if not scores:
        return {}
    highest = max(scores.values())
    return {position: value / highest for position, value in scores.items()} if highest > 0 else {}


async def find(
    index: Index,
    query: str,
    *,
    embeddings: Embeddings | None = None,
    limit: int = 5,
    weight: float = WEIGHT,
    floor: float = MIN_SCORE,
) -> list[Hit]:
    """Разделы под вопрос. Пустой список — значит, в регламентах не нашлось ничего похожего."""
    lexical = search(index, query, limit=max(limit * 4, 20))
    if embeddings is None or index.vectors is None:
        return [hit for hit in lexical if hit.score >= floor][:limit]
    (vector,) = await embeddings.embed([query])
    positions = {section.reference: number for number, section in enumerate(index.sections)}
    lexical_scores = _normalized({positions[hit.section.reference]: hit.score for hit in lexical})
    words = {positions[hit.section.reference]: hit.words for hit in lexical}
    semantic_scores = _normalized(
        {
            position: cosine(vector, index.vectors[position])
            for position in range(len(index.sections))
            if index.vectors[position]
        }
    )
    blended: list[Hit] = []
    for position in set(lexical_scores) | set(semantic_scores):
        score = (1 - weight) * lexical_scores.get(position, 0.0) + weight * semantic_scores.get(position, 0.0)
        blended.append(Hit(index.sections[position], score, words.get(position, ())))
    blended.sort(key=lambda hit: (-hit.score, hit.section.reference))
    best = blended[0].score if blended else 0.0
    return [hit for hit in blended if hit.score >= best * 0.25][:limit] if best else []
