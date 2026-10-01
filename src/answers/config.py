"""Настройки из окружения: где регламенты и индекс, какая модель и нужны ли эмбеддинги."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from answers.llm import Model, NullModel, OllamaModel, OpenAIModel
from answers.search import Embeddings, OllamaEmbeddings

KINDS = ("none", "ollama", "openai")
EMBEDDING_KINDS = ("none", "ollama")


class SettingsError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    corpus: Path = Path("регламенты")
    index_path: Path = Path("индекс.json")
    model_kind: str = "none"
    model_name: str = "qwen3:4b"
    model_url: str = "http://127.0.0.1:11434"
    model_key: str = ""
    embeddings_kind: str = "none"
    embeddings_name: str = "qwen3:4b"
    limit: int = 5
    weight: float = 0.5

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = env if env is not None else os.environ
        model_kind = source.get("ANSWERS_MODEL", "none").strip().lower()
        embeddings_kind = source.get("ANSWERS_EMBEDDINGS", "none").strip().lower()
        if model_kind not in KINDS:
            raise SettingsError(f"ANSWERS_MODEL={model_kind}: бывает {', '.join(KINDS)}")
        if embeddings_kind not in EMBEDDING_KINDS:
            raise SettingsError(f"ANSWERS_EMBEDDINGS={embeddings_kind}: бывает {', '.join(EMBEDDING_KINDS)}")
        settings = cls(
            corpus=Path(source.get("ANSWERS_CORPUS", "регламенты")),
            index_path=Path(source.get("ANSWERS_INDEX", "индекс.json")),
            model_kind=model_kind,
            model_name=source.get("ANSWERS_MODEL_NAME", "qwen3:4b"),
            model_url=source.get("ANSWERS_MODEL_URL", "http://127.0.0.1:11434"),
            model_key=source.get("ANSWERS_MODEL_KEY", ""),
            embeddings_kind=embeddings_kind,
            embeddings_name=source.get("ANSWERS_EMBEDDINGS_NAME", "qwen3:4b"),
            limit=int(source.get("ANSWERS_LIMIT", "5")),
            weight=float(source.get("ANSWERS_WEIGHT", "0.5")),
        )
        if model_kind == "openai" and not settings.model_key:
            raise SettingsError("Для ANSWERS_MODEL=openai нужен ключ в ANSWERS_MODEL_KEY")
        if not 0 <= settings.weight <= 1:
            raise SettingsError("ANSWERS_WEIGHT — доля смысловой близости, от 0 до 1")
        return settings

    def build_model(self) -> Model:
        if self.model_kind == "ollama":
            return OllamaModel(self.model_name, self.model_url)
        if self.model_kind == "openai":
            return OpenAIModel(self.model_name, self.model_url, self.model_key)
        return NullModel()

    def build_embeddings(self) -> Embeddings | None:
        if self.embeddings_kind == "ollama":
            return OllamaEmbeddings(self.embeddings_name, self.model_url)
        return None
