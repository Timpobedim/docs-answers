"""Замер поиска на размеченных вопросах: recall@k, MRR и доля «не нашли».

Без такого замера любая правка поиска — дело вкуса. Набор вопросов лежит рядом с регламентами, и
после каждой правки видно, стало лучше или хуже.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from answers.index import Index
from answers.search import Embeddings, find

LEVELS = (1, 3, 5)


class QuestionsError(ValueError):
    pass


@dataclass(frozen=True)
class Question:
    text: str
    expect: str  # часть ссылки на раздел, например «Отпуска → Сколько дней»


@dataclass(frozen=True)
class Score:
    total: int
    recall: dict[int, float]  # k → доля вопросов, где нужный раздел попал в первые k
    mrr: float
    missed: tuple[str, ...]

    @property
    def found(self) -> float:
        return 1 - len(self.missed) / self.total if self.total else 0.0


def load_questions(path: Path) -> list[Question]:
    try:
        raw = json.loads(path.read_text("utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise QuestionsError(f"Не удалось прочитать вопросы из {path.name}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise QuestionsError(f"{path.name}: ожидается список вопросов")
    questions = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("вопрос") or not item.get("раздел"):
            raise QuestionsError(f"{path.name}: у каждого вопроса нужны поля «вопрос» и «раздел»")
        questions.append(Question(str(item["вопрос"]), str(item["раздел"])))
    return questions


async def measure(
    index: Index,
    questions: Sequence[Question],
    *,
    embeddings: Embeddings | None = None,
    limit: int = 5,
    weight: float = 0.5,
) -> Score:
    hits_at: dict[int, int] = dict.fromkeys(LEVELS, 0)
    reciprocal = 0.0
    missed: list[str] = []
    for question in questions:
        found = await find(index, question.text, embeddings=embeddings, limit=limit, weight=weight)
        places = [
            number
            for number, hit in enumerate(found, start=1)
            if question.expect.lower() in hit.section.reference.lower()
        ]
        if not places:
            missed.append(question.text)
            continue
        best = places[0]
        reciprocal += 1 / best
        for level in LEVELS:
            if best <= level:
                hits_at[level] += 1
    total = len(questions)
    recall = {level: (count / total if total else 0.0) for level, count in hits_at.items()}
    return Score(total, recall, reciprocal / total if total else 0.0, tuple(missed))


def report(score: Score, title: str = "Поиск") -> str:
    lines = [
        f"{title}: вопросов — {score.total}",
        *(f"  recall@{level}: {share:.0%}" for level, share in sorted(score.recall.items())),
        f"  MRR: {score.mrr:.2f}",
        f"  нашли хоть что-то: {score.found:.0%}",
    ]
    if score.missed:
        lines.append("  не нашли: " + "; ".join(score.missed[:5]) + ("…" if len(score.missed) > 5 else ""))
    return "\n".join(lines)
