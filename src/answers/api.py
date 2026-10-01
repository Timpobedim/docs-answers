"""HTTP-слой: спросить, поискать, проверить живость. Ответ всегда со ссылками на разделы."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query

from answers.answer import Answer, extractive, generated
from answers.index import Index
from answers.llm import Model, NullModel
from answers.search import Embeddings, find

MAX_QUESTION = 500


@dataclass
class Service:
    index: Index
    model: Model | None = None
    embeddings: Embeddings | None = None
    limit: int = 5
    weight: float = 0.5

    async def ask(self, question: str) -> Answer:
        hits = await find(self.index, question, embeddings=self.embeddings, limit=self.limit, weight=self.weight)
        if self.model is None or isinstance(self.model, NullModel):
            return extractive(question, hits)
        return await generated(self.model, question, hits)


def as_json(answer: Answer) -> dict[str, Any]:
    return {
        "ответ": answer.text,
        "нашли": answer.found,
        "источник": answer.source,
        "ссылки": [
            {"номер": citation.number, "раздел": citation.reference, "текст": citation.text}
            for citation in answer.citations
        ],
    }


def create_app(service: Service) -> FastAPI:
    app = FastAPI(title="Ответы по регламентам", version="1.0.0")

    def check(question: str) -> str:
        cleaned = " ".join(question.split())
        if not cleaned:
            raise HTTPException(status_code=422, detail="Вопрос пустой")
        if len(cleaned) > MAX_QUESTION:
            raise HTTPException(status_code=422, detail=f"Вопрос длиннее {MAX_QUESTION} символов")
        return cleaned

    @app.get("/health")
    async def health() -> dict[str, Any]:
        model = service.model.name if service.model is not None else "нет модели"
        return {
            "разделов": len(service.index.sections),
            "модель": model,
            "эмбеддинги": service.embeddings.name if service.embeddings else "нет",
        }

    @app.post("/ask")
    async def ask(question: str = Body(..., embed=True, alias="вопрос")) -> dict[str, Any]:
        return as_json(await service.ask(check(question)))

    @app.get("/search")
    async def search(q: str = Query(..., description="что искать")) -> dict[str, Any]:
        hits = await find(
            service.index, check(q), embeddings=service.embeddings, limit=service.limit, weight=service.weight
        )
        return {
            "найдено": len(hits),
            "разделы": [
                {"раздел": hit.section.reference, "балл": round(hit.score, 3), "слова": list(hit.words)} for hit in hits
            ],
        }

    return app
