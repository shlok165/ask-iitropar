"""Builds the messages sent to the LLM: fixed system prompt + short history + CONTEXT and question."""
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from app import config
from app.sources import Source, format_sources

SYSTEM_PROMPT_PATH = config.ROOT / "prompts" / "system_prompt.md"
MAX_HISTORY_CHARS = 2000  # per earlier message; older sources are never resent


def today() -> str:
    return datetime.now(ZoneInfo(config.TIMEZONE)).strftime("%A, %d %B %Y")


@lru_cache(maxsize=4)
def system_prompt(day: str) -> str:
    # Identical for every request on a given day, so providers can serve it from their prompt cache.
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").replace("{today}", day)


def user_message(question: str, sources: list[Source]) -> str:
    context = format_sources(sources) or "(no sources were found for this question)"
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}"


def build_messages(question: str, sources: list[Source], history: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt(today())}]
    for m in (history or [])[-2 * config.HISTORY_TURNS:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": str(m["content"])[:MAX_HISTORY_CHARS]})
    msgs.append({"role": "user", "content": user_message(question, sources)})
    return msgs
