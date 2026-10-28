"""Pollinations: OpenAI-compatible /chat/completions with a Bearer key. Costs pollen."""
import json
import threading

import requests

from app import config
from app.llm.base import LLMError, Provider, Usage, estimate_tokens, messages_tokens

# Pollen per token, from the public price list (GET {base}/models) on 2026-10-22.
# Used only if the live price list can't be fetched.
FALLBACK_PRICES = {
    "openai/gpt-5.4-mini": (0.0000005625, 0.00000005625, 0.000003375),
    "openai/gpt-5.4-nano": (0.00000015, 0.000000015, 0.0000009375),
}
_prices: dict[str, tuple[float, float, float]] = {}
_prices_lock = threading.Lock()


def prices_for(model: str) -> tuple[float, float, float]:
    """(prompt, cached prompt, completion) pollen per token."""
    with _prices_lock:
        if not _prices:
            try:
                r = requests.get(config.POLLINATIONS_BASE_URL.rstrip("/") + "/models", timeout=10)
                for m in r.json().get("data", []):
                    p = m.get("pricing") or {}
                    if "promptTextTokens" in p and "completionTextTokens" in p:
                        entry = (float(p["promptTextTokens"]), float(p.get("promptCachedTokens") or p["promptTextTokens"]),
                                 float(p["completionTextTokens"]))
                        for name in [m["id"], *(m.get("aliases") or [])]:
                            _prices[name] = entry
            except (requests.RequestException, ValueError, KeyError, TypeError):
                pass
    return _prices.get(model) or FALLBACK_PRICES.get(model) or max(FALLBACK_PRICES.values())


class PollinationsProvider(Provider):
    name = "pollinations"
    paid = True

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None):
        super().__init__(model or config.POLLINATIONS_MODEL)
        self.api_key = api_key or config.POLLINATIONS_API_KEY
        self.url = (base_url or config.POLLINATIONS_BASE_URL).rstrip("/") + "/chat/completions"

    def available(self) -> bool:
        return bool(self.api_key)

    def cost(self, prompt: int, cached: int, completion: int) -> float:
        p_in, p_cached, p_out = prices_for(self.model)
        return (prompt - cached) * p_in + cached * p_cached + completion * p_out

    def worst_case_cost(self, messages, max_tokens):
        return self.cost(messages_tokens(messages), 0, max_tokens)

    def generate(self, messages, stream=True):
        if not self.api_key:
            raise LLMError("POLLINATIONS_API_KEY is not set")
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            # GPT-5.4 models reject `temperature`; they take reasoning_effort, and hidden reasoning
            # tokens count against max_completion_tokens.
            "reasoning_effort": config.LLM_REASONING_EFFORT,
            "max_completion_tokens": config.LLM_MAX_TOKENS,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        try:
            r = requests.post(self.url, json=body, stream=stream, timeout=(10, 120),
                              headers={"Authorization": f"Bearer {self.api_key}"})
        except requests.RequestException as e:
            raise LLMError(f"Pollinations unreachable: {e.__class__.__name__}") from e
        if r.status_code != 200:
            raise self._error(r)

        text, usage = [], None
        if stream:
            for raw in r.iter_lines():
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        text.append(piece)
                        yield piece
        else:
            data = r.json()
            usage = data.get("usage")
            piece = data["choices"][0]["message"].get("content") or ""
            text.append(piece)
            yield piece
        yield self._usage(usage, messages, "".join(text))

    def _usage(self, u: dict | None, messages, text: str) -> Usage:
        if u:
            prompt, completion = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            return Usage(prompt, completion, cached, self.cost(prompt, cached, completion))
        # No usage reported: assume the full completion budget was spent (reasoning tokens are invisible).
        prompt, completion = messages_tokens(messages), max(config.LLM_MAX_TOKENS, estimate_tokens(text))
        return Usage(prompt, completion, 0, self.cost(prompt, 0, completion), estimated=True)

    @staticmethod
    def _error(r: requests.Response) -> LLMError:
        try:
            err = r.json().get("error")
            msg = err.get("message") if isinstance(err, dict) else str(err or r.text)
        except ValueError:
            msg = r.text
        credit_words = ("pollen", "balance", "credit", "budget", "quota")
        out_of_credit = r.status_code == 402 or (r.status_code in (403, 429)
                                                 and any(w in msg.lower() for w in credit_words))
        return LLMError(f"Pollinations HTTP {r.status_code}: {msg[:300]}", r.status_code, out_of_credit)
