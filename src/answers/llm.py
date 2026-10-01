"""Модель для связного ответа — необязательная деталь.

Без модели сервис отвечает выдержками из регламентов, и это уже полезно. С моделью ответ становится
связным, но источником остаются те же найденные фрагменты.

Модель подключается отдельно от поиска: без поднятого Ollama поиск всё равно работает.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

import httpx

TIMEOUT = httpx.Timeout(180.0, connect=5.0)
THINKING = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class ModelError(RuntimeError):
    pass


class Model(Protocol):
    name: str

    async def complete(self, system: str, user: str) -> str: ...


class NullModel:
    """Модель не настроена: отвечаем выдержками из регламентов."""

    name = "нет модели"

    async def complete(self, system: str, user: str) -> str:
        raise ModelError("Модель не настроена")


def clean(text: str) -> str:
    """Убираем рассуждения модели: это не ответ сотруднику.

    Рассуждающие модели вроде qwen3 присылают их по-разному: то парой тегов <think>…</think>, то
    просто текстом, который заканчивается одиноким </think> — второй случай встретился на живом
    прогоне, и без этой обработки рассуждение уезжало прямо в ответ.
    """
    without_pairs = THINKING.sub(" ", text)
    after_thought = without_pairs.rsplit("</think>", 1)[-1]
    return " ".join(after_thought.replace("<think>", " ").split())


class OllamaModel:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", client: httpx.AsyncClient | None = None):
        self.name = f"ollama:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client

    async def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        data = await _post(self._client, f"{self.base_url}/api/chat", payload, {})
        return clean(str((data.get("message") or {}).get("content", "")))


class OpenAIModel:
    def __init__(self, model: str, base_url: str, api_key: str, client: httpx.AsyncClient | None = None):
        self.name = f"openai:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = client

    async def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        data = await _post(
            self._client, f"{self.base_url}/v1/chat/completions", payload, {"Authorization": f"Bearer {self.api_key}"}
        )
        choices = data.get("choices") or [{}]
        return clean(str((choices[0].get("message") or {}).get("content", "")))


async def _post(
    client: httpx.AsyncClient | None, url: str, payload: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    own = client is None
    session = client or httpx.AsyncClient(timeout=TIMEOUT)
    try:
        response = await session.post(url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise ModelError(f"Модель ответила {response.status_code}: {response.text[:200]}")
        return dict(response.json())
    except httpx.HTTPError as exc:
        raise ModelError(f"Модель недоступна: {exc}") from exc
    finally:
        if own:
            await session.aclose()
