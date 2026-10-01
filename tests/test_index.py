from __future__ import annotations

from pathlib import Path

from answers.documents import Section
from answers.index import Index, build, load, save, search, stem, tokenize


def test_stemmer_joins_word_forms() -> None:
    assert len({stem(word) for word in ("отпуска", "отпуску", "отпуском", "отпуски")}) == 1
    assert stem("пропуск") == "пропуск"
    assert stem("дом") == "дом"  # короткое слово не режем


def test_tokenizer_drops_noise() -> None:
    assert tokenize("Как и в 2026 году, это же просто!") == [tokenize("2026")[0], "год", "прост"]
    assert tokenize("") == []


def test_search_finds_the_right_section(index: Index) -> None:
    best = search(index, "сколько дней отпуска положено", limit=3)
    assert best
    assert "Сколько дней отпуска положено" in best[0].section.reference
    assert best[0].words

    flash = search(index, "можно ли пользоваться личной флешкой", limit=3)
    assert "Внешние носители" in flash[0].section.reference

    assert search(index, "как приготовить борщ", limit=3) == []


def test_rare_words_matter_more(index: Index) -> None:
    hits = search(index, "суточные", limit=3)
    assert "Суточные и проживание" in hits[0].section.reference
    assert hits[0].score > 0


def test_index_survives_save_and_load(index: Index, tmp_path: Path) -> None:
    path = tmp_path / "индекс.json"
    save(index, path)
    restored = load(path)
    assert [section.reference for section in restored.sections] == [section.reference for section in index.sections]
    assert [hit.section.reference for hit in search(restored, "гостевой пропуск")] == [
        hit.section.reference for hit in search(index, "гостевой пропуск")
    ]


def test_empty_index_answers_nothing() -> None:
    empty = build([])
    assert search(empty, "что угодно") == []
    assert build([Section("a.md", ("Раздел",), "Текст про закупки", 1)]).average_length > 0
