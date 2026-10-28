"""Web-search fallback: search -> fetch top pages -> clean -> best passages -> redact -> 'web' sources.

The search engine sits behind SearchProvider, so a paid API (e.g. Exa) can replace the free ddgs later:
write a class with .search(query, max_results) -> list[SearchResult] and register it in PROVIDERS.
"""
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
from bs4 import BeautifulSoup
from pypdf import PdfReader

from app import config
from app.search import QUERY_PREFIX
from app.sources import Source
from app.state import local_day
from scraper.common import clean_text, get, redact, robots_allowed

MAX_PAGE_BYTES = 3_000_000
PASSAGE_WORDS, PASSAGE_STRIDE = 180, 140
PASSAGES_PER_PAGE = 2
MIN_PAGE_WORDS = 60  # less than this and we fall back to the search snippet
FETCH_TIMEOUT = 10
_log_lock = threading.Lock()


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class DDGSSearch:
    """Free metasearch via the ddgs package (DuckDuckGo and others)."""
    name = "ddgs"

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException
        try:
            rows = DDGS().text(query, max_results=max_results, region="in-en", safesearch="moderate")
        except DDGSException:  # includes "No results found" and rate limiting
            return []
        return [SearchResult(r.get("title") or "", r.get("href") or "", r.get("body") or "") for r in rows if r.get("href")]


PROVIDERS = {"ddgs": DDGSSearch}


def get_search_provider(name: str | None = None):
    name = name or config.WEB_SEARCH_PROVIDER
    if name not in PROVIDERS:
        raise ValueError(f"Unknown WEB_SEARCH_PROVIDER '{name}' (available: {', '.join(PROVIDERS)})")
    return PROVIDERS[name]()


def host_matches(url: str, domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in domains)


def build_query(question: str, scope: str) -> str:
    if scope == "official":
        return f"{question} (" + " OR ".join(f"site:{d}" for d in config.WEB_ALLOWED_DOMAINS) + ")"
    return question if re.search(r"\b(iit\s*ropar|iitrpr|rupnagar)\b", question, re.I) else f"{question} IIT Ropar"


def fetch_text(url: str) -> str:
    """Main text of a page or PDF ('' if it can't be fetched)."""
    if not robots_allowed(url):
        return ""
    r = get(url, timeout=FETCH_TIMEOUT, stream=True, delay=0)
    if r is None or r.status_code != 200:
        return ""
    data = b""
    for chunk in r.iter_content(65536):
        data += chunk
        if len(data) > MAX_PAGE_BYTES:
            break
    ctype = r.headers.get("content-type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            return clean_text("\n".join(p.extract_text() or "" for p in reader.pages[:20]))
        except Exception:
            return ""
    if "html" not in ctype and "text" not in ctype:
        return ""
    soup = BeautifulSoup(data, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe",
                     "button", "template"]):
        tag.decompose()
    return clean_text(soup.get_text("\n"))


def passages(text: str) -> list[str]:
    """Overlapping windows that cover the whole text, including its end."""
    words = text.split()
    if len(words) <= PASSAGE_WORDS:
        return [" ".join(words)] if words else []
    starts = list(range(0, len(words) - PASSAGE_WORDS + 1, PASSAGE_STRIDE))
    if starts[-1] + PASSAGE_WORDS < len(words):
        starts.append(len(words) - PASSAGE_WORDS)
    return [" ".join(words[i:i + PASSAGE_WORDS]) for i in starts]


class WebSearcher:
    def __init__(self, model, provider=None, fetch=fetch_text):
        self.model = model  # the same bge embedding model the index uses, to pick relevant passages
        self.provider = provider or get_search_provider()
        self.fetch = fetch

    def search(self, question: str, scope: str) -> list[SearchResult]:
        wanted = config.WEB_SEARCH_MAX_RESULTS
        results = self.provider.search(build_query(question, scope), max_results=wanted * 2 + 2)
        keep, seen = [], set()
        for r in results:
            if r.url in seen or host_matches(r.url, config.WEB_BLOCKED_DOMAINS):
                continue
            if scope == "official" and not host_matches(r.url, config.WEB_ALLOWED_DOMAINS):
                continue
            seen.add(r.url)
            keep.append(r)
        return keep[:wanted]

    def gather(self, question: str, scope: str) -> list[Source]:
        results = self.search(question, scope)
        if not results:
            return []
        with ThreadPoolExecutor(max_workers=len(results)) as pool:
            texts = list(pool.map(lambda r: self._safe_fetch(r.url), results))

        q = self.model.encode([QUERY_PREFIX + question], normalize_embeddings=True)[0]
        scored = []
        for r, text in zip(results, texts):
            note = ""
            if len(text.split()) < MIN_PAGE_WORDS:
                # JavaScript-only sites (e.g. the fest sites) come back nearly empty without a browser;
                # the search engine's snippet is then the best text we have.
                text = "\n".join(x for x in (r.snippet, text) if x)
                note = "(search-result snippet only; the page itself could not be read)\n"
            chunks = passages(text)
            if not chunks:
                continue
            emb = self.model.encode(chunks, normalize_embeddings=True)
            sims = emb @ q
            best = np.argsort(-sims)[:PASSAGES_PER_PAGE]
            body = note + "\n[...]\n".join(chunks[i] for i in sorted(best))
            scored.append((float(sims[best[0]]), Source(r.title or r.url, r.url, "web", local_day(), redact(body))))
        return [s for _, s in sorted(scored, key=lambda x: -x[0])]

    def _safe_fetch(self, url: str) -> str:
        try:
            return self.fetch(url)
        except Exception:
            return ""


def log_web_query(question: str, query_used: str, scope: str, trigger: str, sources: list[Source], provider: str) -> None:
    """Every web-answered question goes to a file, to find gaps in the official data."""
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": redact(question), "searched": redact(query_used),
              "scope": scope, "trigger": trigger, "llm": provider, "results": len(sources),
              "urls": [s.url for s in sources]}
    config.LOG_WEB_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, config.LOG_WEB_QUERIES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
