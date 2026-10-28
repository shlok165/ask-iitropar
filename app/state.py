"""Small persistent state in SQLite: answer cache, per-call usage/spend, and an in-memory rate limiter."""
import hashlib
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    key TEXT PRIMARY KEY, question TEXT, answer TEXT, meta TEXT, created REAL, expires REAL);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, day TEXT, provider TEXT, model TEXT,
    prompt_tokens INTEGER, cached_tokens INTEGER, completion_tokens INTEGER, cost REAL, estimated INTEGER);
CREATE INDEX IF NOT EXISTS usage_day ON usage(day, provider);
"""


def local_day() -> str:
    return datetime.now(ZoneInfo(config.TIMEZONE)).strftime("%Y-%m-%d")


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", question.lower())).strip()


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.STATE_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            with conn:  # commits on success
                yield conn
        finally:
            conn.close()

    # --- answer cache ---------------------------------------------------------------
    @staticmethod
    def cache_key(question: str, provider: str, model: str) -> str:
        return hashlib.sha1(f"{normalize(question)}|{provider}|{model}".encode()).hexdigest()

    def get_answer(self, key: str) -> tuple[str, dict] | None:
        with self._conn() as c:
            row = c.execute("SELECT answer, meta FROM answers WHERE key = ? AND expires > ?",
                            (key, time.time())).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def put_answer(self, key: str, question: str, answer: str, meta: dict, ttl_seconds: float) -> None:
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?, ?)",
                      (key, question, answer, json.dumps(meta, ensure_ascii=False), now, now + ttl_seconds))

    # --- spend ----------------------------------------------------------------------
    def record_usage(self, provider: str, model: str, usage) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO usage (ts, day, provider, model, prompt_tokens, cached_tokens, completion_tokens, "
                      "cost, estimated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (time.time(), local_day(), provider, model, usage.prompt_tokens, usage.cached_tokens,
                       usage.completion_tokens, usage.cost, int(usage.estimated)))

    def spent_today(self, provider: str) -> float:
        with self._conn() as c:
            (total,) = c.execute("SELECT COALESCE(SUM(cost), 0) FROM usage WHERE day = ? AND provider = ?",
                                 (local_day(), provider)).fetchone()
        return total


class RateLimiter:
    """Sliding one-minute window per client (IP address)."""

    def __init__(self, per_minute: int | None = None):
        self.per_minute = per_minute or config.RATE_LIMIT_PER_MINUTE
        self.hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self.hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True
