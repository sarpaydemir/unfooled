from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class AIResponse:
    payload: dict[str, Any]
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    request_id: str | None = None


class AIProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, schema: dict[str, Any]) -> AIResponse:
        raise NotImplementedError


class OpenRouterProvider(AIProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: int = 120,
        max_attempts: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or os.environ.get("RESEARCH_AI_MODEL", "openai/gpt-5.2")
        self.temperature = (
            float(os.environ.get("RESEARCH_AI_TEMPERATURE", "0.2"))
            if temperature is None
            else float(temperature)
        )
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        self.system_prompt = system_prompt or (
            "You design causal, testable quantitative research candidates. "
            "Never use future data. Return only the required structured output."
        )
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    def generate(self, prompt: str, schema: dict[str, Any]) -> AIResponse:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "research_candidate",
                    "strict": True,
                    "schema": schema,
                },
            },
            "temperature": self.temperature,
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "Sarp Local Research Factory",
            },
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read())
                content = raw["choices"][0]["message"]["content"]
                payload = json.loads(content) if isinstance(content, str) else content
                usage = raw.get("usage", {})
                return AIResponse(
                    payload=payload,
                    provider="openrouter",
                    model=raw.get("model", self.model),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    cost_usd=float(usage.get("cost", 0.0) or 0.0),
                    request_id=raw.get("id"),
                )
            except (HTTPError, URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < self.max_attempts:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(f"OpenRouter request failed after {self.max_attempts} attempts") from last_error


class StaticProvider(AIProvider):
    """Deterministic provider for local integration tests and dry demos."""

    def __init__(
        self,
        payload: dict[str, Any],
        model: str = "static-v1",
        provider: str = "static",
    ) -> None:
        self.payload = payload
        self.model = model
        self.provider = provider

    def generate(self, prompt: str, schema: dict[str, Any]) -> AIResponse:
        return AIResponse(self.payload, self.provider, self.model, 0, 0, 0.0)
