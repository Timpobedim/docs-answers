from __future__ import annotations

import json
from pathlib import Path

import pytest

from answers.index import Index
from answers.quality import Question, QuestionsError, load_questions, measure, report


async def test_search_quality_on_marked_questions(index: Index, questions: list[Question]) -> None:
    score = await measure(index, questions)
    assert score.total == len(questions)
    assert score.recall[5] >= 0.75  # порог держим с запасом: ниже — значит, поиск сломали
    assert score.recall[1] >= 0.5
    assert score.mrr >= 0.55
    printed = report(score)
    assert "recall@5" in printed
    assert "MRR" in printed


async def test_missed_questions_are_named(index: Index) -> None:
    score = await measure(index, [Question("как приготовить борщ", "Рецепты → Борщ")])
    assert score.missed == ("как приготовить борщ",)
    assert score.found == 0.0
    assert "не нашли" in report(score)


def test_questions_file_is_checked(tmp_path: Path) -> None:
    path = tmp_path / "вопросы.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(QuestionsError, match="список вопросов"):
        load_questions(path)
    path.write_text(json.dumps([{"вопрос": "а раздел?"}], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(QuestionsError, match="«вопрос» и «раздел»"):
        load_questions(path)
    with pytest.raises(QuestionsError, match="Не удалось прочитать"):
        load_questions(tmp_path / "нет-файла.json")
