"""Ollama: free local models through the native /api/chat endpoint.

The native API (not Ollama's OpenAI-compatible one) is used because it lets us raise num_ctx per
request; Ollama's default context window is smaller than our prompts and silently truncates them.
"""
import json

import requests

from app import config
from app.llm.base import LLMError, Provider, Usage


class OllamaProvider(Provider):
    name = "ollama"
    paid = False

    def __init__(self, model: str | None = None, base_url: str | None = None):
        super().__init__(model or config.OLLAMA_MODEL)
        self.base = (base_url or config.OLLAMA_BASE_URL).rstrip("/")

    def available(self) -> bool:
        try:
            tags = requests.get(self.base + "/api/tags", timeout=2).json().get("models", [])
            return any(m.get("name") == self.model or m.get("model") == self.model for m in tags)
        except (requests.RequestException, ValueError):
            return False

    def generate(self, messages, stream=True):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": "30m",
            "options": {"num_ctx": config.OLLAMA_NUM_CTX, "num_predict": config.LLM_MAX_TOKENS,
                        "temperature": 0.2},
        }
        try:
            r = requests.post(self.base + "/api/chat", json=body, stream=stream, timeout=(5, 300))
        except requests.RequestException as e:
            raise LLMError(f"Ollama unreachable at {self.base}: {e.__class__.__name__}") from e
        if r.status_code != 200:
            raise LLMError(f"Ollama HTTP {r.status_code}: {r.text[:300]}", r.status_code)

        final = {}
        lines = r.iter_lines() if stream else [r.content]
        for raw in lines:
            if not raw:
                continue
            chunk = json.loads(raw)
            if chunk.get("error"):
                raise LLMError(f"Ollama: {chunk['error']}")
            piece = (chunk.get("message") or {}).get("content")
            if piece:
                yield piece
            if chunk.get("done"):
                final = chunk
        yield Usage(final.get("prompt_eval_count", 0), final.get("eval_count", 0))
