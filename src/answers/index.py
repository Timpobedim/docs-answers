"""Поисковый индекс: разбор текста на слова, лёгкий стеммер и BM25 — без внешних библиотек.

Для поиска по регламентам небольшой компании этого достаточно: тысячи разделов, а не миллионы.
Индекс — обычный JSON-файл: его видно глазами, он ложится в репозиторий и не тянет за собой базу
векторов. Эмбеддинги подключаются отдельно и не обязательны (см. `search.py`).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from answers.documents import Section

K1 = 1.5
B = 0.75
WORD = re.compile(r"[а-яa-z0-9]+")
_STOPWORDS = """и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот
    от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам
    ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз
    тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее
    сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про
    всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им
    более всегда конечно всю между это"""
STOPWORDS = frozenset(_STOPWORDS.split())
# Лёгкий стеммер: отрезаем самое длинное знакомое окончание, если остаётся не меньше трёх букв.
# Это не Snowball, но «отпуска», «отпуску» и «отпуском» сводятся к «отпуск» — для поиска хватает.
# «ка», «ку», «ки» сюда не входят намеренно: иначе «отпуска» превращается в «отпус», а «флешки» и
# «флешкой» перестают сходиться в одно слово.
ENDINGS = (
    "ированиями", "ированиях", "ированием", "ированию", "ирования",
    "ями", "ами", "ией", "иях", "иям", "ием", "его", "ого", "ему", "ому", "ев", "ов", "ие", "ье", "ая", "яя",
    "ое", "ые", "ий", "ый", "ой", "ем", "им", "ым", "ом", "их", "ых", "ую", "юю", "ах", "ях", "ью", "ия", "ья",
    "ей", "ть", "ся", "сь", "ет", "ут", "ют", "ат", "ят", "ешь", "ишь", "ла", "ло", "ли",
    "л", "а", "е", "и", "й", "о", "у", "ы", "ь", "ю", "я",
)  # fmt: skip
MIN_STEM = 3


def stem(word: str) -> str:
    for ending in sorted(ENDINGS, key=len, reverse=True):
        if word.endswith(ending) and len(word) - len(ending) >= MIN_STEM:
            return word[: -len(ending)]
    return word


def tokenize(text: str) -> list[str]:
    words = WORD.findall(text.lower().replace("ё", "е"))
    return [stem(word) for word in words if word not in STOPWORDS and len(word) > 1]


@dataclass
class Index:
    sections: list[Section]
    documents: list[Counter[str]]  # слова каждого раздела
    lengths: list[int]
    frequencies: Counter[str]  # в скольких разделах встречается слово
    vectors: list[list[float]] | None = None  # эмбеддинги разделов, если их считали

    @property
    def average_length(self) -> float:
        return sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

    def idf(self, word: str) -> float:
        total, seen = len(self.sections), self.frequencies.get(word, 0)
        return math.log(1 + (total - seen + 0.5) / (seen + 0.5))


@dataclass(frozen=True)
class Hit:
    section: Section
    score: float
    words: tuple[str, ...] = ()  # слова запроса, которые нашлись в разделе


def build(sections: list[Section], vectors: list[list[float]] | None = None) -> Index:
    documents = [Counter(tokenize(f"{section.title} {section.text}")) for section in sections]
    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(document.keys())
    return Index(sections, documents, [sum(document.values()) for document in documents], frequencies, vectors)


def search(index: Index, query: str, limit: int = 5) -> list[Hit]:
    """BM25: чем реже слово в регламентах и чем чаще в разделе, тем выше вклад."""
    words = tokenize(query)
    if not words or not index.sections:
        return []
    average = index.average_length or 1.0
    hits: list[Hit] = []
    for position, document in enumerate(index.documents):
        score = 0.0
        found: list[str] = []
        length = index.lengths[position] or 1
        for word in set(words):
            count = document.get(word, 0)
            if not count:
                continue
            found.append(word)
            score += index.idf(word) * count * (K1 + 1) / (count + K1 * (1 - B + B * length / average))
        if score > 0:
            hits.append(Hit(index.sections[position], score, tuple(sorted(found))))
    hits.sort(key=lambda hit: (-hit.score, hit.section.reference))
    return hits[:limit]


def save(index: Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "sections": [
            {
                "document": section.document,
                "path": list(section.path),
                "text": section.text,
                "number": section.number,
            }
            for section in index.sections
        ],
        "vectors": index.vectors,
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load(path: Path) -> Index:
    data = json.loads(path.read_text("utf-8"))
    sections = [
        Section(str(item["document"]), tuple(item["path"]), str(item["text"]), int(item["number"]))
        for item in data["sections"]
    ]
    return build(sections, data.get("vectors"))
