"""Ответ с цитатами: либо выдержки из регламентов, либо связный текст модели по тем же фрагментам.

Правило одно: отвечаем только тем, что нашлось в регламентах. Если поиск ничего не дал — так и
говорим. Если модель ответила, не сославшись ни на один фрагмент, её ответ не показываем: скорее
всего, она придумала его сама.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from answers.index import Hit
from answers.llm import Model, ModelError, clean

SENTENCE = re.compile(r"(?<=[.!?])\s+")
CITATION = re.compile(r"\[(\d+)\]")
SENTENCES_IN_QUOTE = 2
QUOTES = 2  # сколько разделов цитировать
CLOSE_ENOUGH = 0.6  # второй раздел цитируем, только если он почти так же подходит, как первый
NOTHING = "В регламентах такого не нашлось. Уточните вопрос или спросите у кадровой службы."
SYSTEM = (
    "Ты отвечаешь сотрудникам по внутренним регламентам компании. Отвечай только тем, что есть во "
    "фрагментах. Каждое утверждение подкрепляй ссылкой в квадратных скобках: [1], [2]. Если во "
    "фрагментах ответа нет, ответь ровно: «В регламентах ответа нет». Не придумывай пункты и сроки, "
    "которых нет в тексте. Отвечай по-русски, тремя-четырьмя предложениями."
)


@dataclass(frozen=True)
class Citation:
    number: int
    reference: str
    text: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    found: bool
    source: str  # «регламенты» или «модель»


def citations_of(hits: Sequence[Hit]) -> tuple[Citation, ...]:
    return tuple(Citation(number, hit.section.reference, hit.section.text) for number, hit in enumerate(hits, start=1))


def _quote(text: str, question: str) -> str:
    """Предложения раздела, где больше всего слов вопроса, — коротко и по делу."""
    from answers.index import tokenize

    wanted = set(tokenize(question))
    sentences = [sentence.strip() for sentence in SENTENCE.split(text) if sentence.strip()]
    if not sentences:
        return text
    scored = sorted(
        enumerate(sentences),
        key=lambda pair: (-len(wanted & set(tokenize(pair[1]))), pair[0]),
    )
    chosen = sorted(position for position, _ in scored[:SENTENCES_IN_QUOTE])
    return " ".join(sentences[position] for position in chosen)


def extractive(question: str, hits: Sequence[Hit]) -> Answer:
    """Ответ без модели: выдержки из найденных разделов.

    Цитируем только те разделы, которые подходят почти так же хорошо, как первый: иначе во втором
    абзаце оказывается что-нибудь про отпуск по уходу за ребёнком, просто потому что там встретилось
    словосочетание «рабочее место».
    """
    citations = citations_of(hits)
    if not citations:
        return Answer(NOTHING, (), found=False, source="регламенты")
    best = hits[0].score or 1.0
    quoted = [
        citation
        for citation, hit in zip(citations[:QUOTES], hits[:QUOTES], strict=True)
        if hit.score >= best * CLOSE_ENOUGH
    ]
    lines = [f"{_quote(citation.text, question)} [{citation.number}]" for citation in quoted]
    return Answer(" ".join(lines), citations, found=True, source="регламенты")


def fragments(hits: Sequence[Hit]) -> str:
    return "\n\n".join(
        f"[{number}] {hit.section.reference}\n{hit.section.text}" for number, hit in enumerate(hits, start=1)
    )


async def generated(model: Model, question: str, hits: Sequence[Hit]) -> Answer:
    """Ответ модели по найденным фрагментам. Без ссылок на фрагменты ответ не принимается."""
    citations = citations_of(hits)
    if not citations:
        return Answer(NOTHING, (), found=False, source="регламенты")
    try:
        raw = await model.complete(SYSTEM, f"Вопрос: {question}\n\nФрагменты регламентов:\n{fragments(hits)}")
    except ModelError:
        return extractive(question, hits)
    text = clean(raw)  # рассуждения модели убираем здесь: это касается любой модели, не только Ollama
    if "в регламентах ответа нет" in text.lower():
        return Answer(NOTHING, (), found=False, source="модель")  # без ссылок: ссылаться не на что
    used = {int(number) for number in CITATION.findall(text) if 0 < int(number) <= len(citations)}
    if not used:  # модель ответила «из головы» — показываем выдержки, а не её фантазию
        return extractive(question, hits)
    return Answer(text, tuple(citation for citation in citations if citation.number in used), True, "модель")


def show(answer: Answer) -> str:
    lines = [answer.text]
    if answer.citations:
        lines.append("")
        lines.extend(f"[{citation.number}] {citation.reference}" for citation in answer.citations)
    return "\n".join(lines)
