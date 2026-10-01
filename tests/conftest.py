from __future__ import annotations

from pathlib import Path

import pytest

from answers.documents import Section, load
from answers.index import Index, build
from answers.quality import Question, load_questions

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="session")
def sections() -> list[Section]:
    return load(EXAMPLES / "регламенты")


@pytest.fixture(scope="session")
def index(sections: list[Section]) -> Index:
    return build(sections)


@pytest.fixture(scope="session")
def questions() -> list[Question]:
    return load_questions(EXAMPLES / "вопросы.json")
