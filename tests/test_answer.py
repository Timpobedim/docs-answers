from __future__ import annotations

import pytest

from answers.answer import NOTHING, extractive, generated, show
from answers.documents import Section
from answers.index import Hit
from answers.llm import Model, ModelError

SECTION = Section(
    "отпуска.md",
    ("Отпуска", "Сколько дней отпуска положено"),
    "Ежегодный оплачиваемый отпуск составляет 28 календарных дней. "
    "Сотрудникам с ненормированным рабочим днём добавляется 3 календарных дня. "
    "Право на первый отпуск возникает через 6 месяцев работы.",
    1,
)
HITS = (Hit(SECTION, 4.2, ("отпуск", "ден")),)


class Scripted:
    name = "сценарий"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.asked: list[str] = []

    async def complete(self, system: str, user: str) -> str:
        self.asked.append(user)
        return self.answer


class Broken:
    name = "сломанная"

    async def complete(self, system: str, user: str) -> str:
        raise ModelError("модель недоступна")


def test_extractive_answer_quotes_the_right_sentences() -> None:
    answer = extractive("сколько дней отпуска", HITS)
    assert answer.found
    assert answer.source == "регламенты"
    assert "28 календарных дней" in answer.text
    assert answer.text.endswith("[1]")
    assert answer.citations[0].reference == "отпуска.md · Отпуска → Сколько дней отпуска положено"


def test_weak_second_section_is_listed_but_not_quoted() -> None:
    weak = Section("отпуска.md", ("Отпуска", "Отпуск по уходу за ребёнком"), "За сотрудником сохраняется место.", 1)
    answer = extractive("сколько дней отпуска", (HITS[0], Hit(weak, 0.9, ("мест",))))
    assert "28 календарных дней" in answer.text
    assert "сохраняется место" not in answer.text  # цитируем только то, что действительно подходит
    assert [citation.number for citation in answer.citations] == [1, 2]  # но показываем обе ссылки


def test_nothing_found() -> None:
    answer = extractive("как приготовить борщ", ())
    assert not answer.found
    assert answer.text == NOTHING
    assert answer.citations == ()


async def test_model_answer_keeps_only_used_citations() -> None:
    model: Model = Scripted("Отпуск — 28 календарных дней [1].")
    answer = await generated(model, "сколько дней отпуска", HITS)
    assert answer.source == "модель"
    assert answer.found
    assert [citation.number for citation in answer.citations] == [1]
    assert "Фрагменты регламентов" in model.asked[0]  # type: ignore[attr-defined]


async def test_answer_without_citations_is_not_trusted() -> None:
    answer = await generated(Scripted("Кажется, отпуск бывает 30 дней."), "сколько дней отпуска", HITS)
    assert answer.source == "регламенты"  # откатились к выдержкам из текста
    assert "28 календарных дней" in answer.text


async def test_model_may_say_there_is_no_answer() -> None:
    answer = await generated(Scripted("В регламентах ответа нет"), "какая погода", HITS)
    assert not answer.found
    assert answer.text == NOTHING
    assert answer.citations == ()  # ссылаться не на что — и ссылок быть не должно


async def test_reasoning_of_the_model_is_cut_off() -> None:
    """На живом прогоне qwen3 прислала рассуждение без открывающего тега, только с закрывающим."""
    thinking = "Посмотрю фрагменты. В [1] написано про 28 дней. </think> Отпуск — 28 календарных дней [1]."
    answer = await generated(Scripted(thinking), "сколько дней отпуска", HITS)
    assert answer.text == "Отпуск — 28 календарных дней [1]."
    assert "Посмотрю фрагменты" not in answer.text


async def test_broken_model_falls_back_to_quotes() -> None:
    answer = await generated(Broken(), "сколько дней отпуска", HITS)
    assert answer.source == "регламенты"
    assert answer.found


def test_show_prints_references() -> None:
    printed = show(extractive("сколько дней отпуска", HITS))
    assert "[1] отпуска.md · Отпуска → Сколько дней отпуска положено" in printed


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_still_answers_something(question: str) -> None:
    assert extractive(question, ()).text == NOTHING
