from __future__ import annotations

from pathlib import Path

import pytest
from conftest import EXAMPLES
from fastapi.testclient import TestClient

from answers.api import Service, create_app
from answers.cli import main
from answers.index import Index


@pytest.fixture
def client(index: Index) -> TestClient:
    return TestClient(create_app(Service(index)))


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["разделов"] > 20
    assert body["модель"] == "нет модели"


def test_ask_returns_answer_with_references(client: TestClient) -> None:
    response = client.post("/ask", json={"вопрос": "сколько дней отпуска положено"})
    assert response.status_code == 200
    body = response.json()
    assert body["нашли"] is True
    assert "28 календарных дней" in body["ответ"]
    assert body["ссылки"][0]["раздел"].startswith("отпуска.md")


def test_ask_about_nothing(client: TestClient) -> None:
    body = client.post("/ask", json={"вопрос": "как приготовить борщ"}).json()
    assert body["нашли"] is False
    assert body["ссылки"] == []


def test_empty_question_is_rejected(client: TestClient) -> None:
    assert client.post("/ask", json={"вопрос": "   "}).status_code == 422
    assert client.post("/ask", json={"вопрос": "а" * 600}).status_code == 422


def test_search_endpoint(client: TestClient) -> None:
    body = client.get("/search", params={"q": "гостевой пропуск"}).json()
    assert body["найдено"] >= 1
    assert "Гостевой пропуск" in body["разделы"][0]["раздел"]


def test_cli_index_ask_quality(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index_path = tmp_path / "индекс.json"
    assert main(["index", "--corpus", str(EXAMPLES / "регламенты"), "--out", str(index_path)]) == 0
    assert "Разделов:" in capsys.readouterr().out
    assert main(["ask", "сколько дней отпуска положено", "--index", str(index_path)]) == 0
    printed = capsys.readouterr().out
    assert "28 календарных дней" in printed
    assert "отпуска.md" in printed
    assert main(["quality", "--questions", str(EXAMPLES / "вопросы.json"), "--index", str(index_path)]) == 0
    assert "recall@5" in capsys.readouterr().out


def test_cli_demo(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo"]) == 0
    printed = capsys.readouterr().out
    assert "Разделов в индексе:" in printed
    assert "В регламентах такого не нашлось" in printed  # вопрос про погоду
    assert "recall@5" in printed


def test_missing_index_is_explained(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "вопрос", "--index", str(tmp_path / "нет.json")]) == 2
    assert "соберите его" in capsys.readouterr().err


def test_bad_settings_are_explained(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("ANSWERS_MODEL", "магия")
    assert main(["demo"]) == 2
    assert "ANSWERS_MODEL" in capsys.readouterr().err
