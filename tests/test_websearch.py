"""Web-search fallback tests: fake search engine, fake page fetcher, tiny fake embedder. No network."""
import json
import re
import zlib

import numpy as np
import pytest

from app import config
from app.answer import AnswerEngine
from app.llm.fake import FakeProvider
from app.state import RateLimiter, Store
from app.tables import TableStore
from app.websearch import SearchResult, WebSearcher, build_query, host_matches
from tests.test_answer import StubBuilder


class BagOfWords:
    """Stands in for the bge model: hashed word counts, normalized."""

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class FakeSearch:
    name = "fake"

    def __init__(self, results):
        self.results, self.queries = results, []

    def search(self, query, max_results):
        self.queries.append(query)
        return self.results[:max_results]


RESULTS = [
    SearchResult("Zeitgeist on Instagram", "https://www.instagram.com/zeitgeist.iitrpr/", "photos"),
    SearchResult("Zeitgeist 2026", "https://www.zeitgeist.org.in/", "Zeitgeist 2026 runs 30 Oct to 1 Nov"),
    SearchResult("Random blog", "https://someblog.example.com/zeitgeist", "a blog post"),
    SearchResult("Hostel", "https://hostel.iitrpr.ac.in/rules", "hostel rules"),
    SearchResult("Fake iitrpr", "https://iitrpr.ac.in.evil.com/x", "lookalike domain"),
]
PAGES = {
    "https://www.zeitgeist.org.in/": "Menu Home Events. " + "filler words here. " * 80
    + "Zeitgeist 2026, the cultural fest, runs from 30 October to 1 November 2026. Contact 98765 43210.",
    "https://hostel.iitrpr.ac.in/rules": "Hostel rules: visitors leave by 10 pm. " * 5,
    "https://someblog.example.com/zeitgeist": "I went to Zeitgeist and it was fun. " * 5,
}


def searcher(results=RESULTS):
    return WebSearcher(BagOfWords(), provider=FakeSearch(results), fetch=lambda url: PAGES.get(url, ""))


def test_host_matching_handles_subdomains_and_lookalikes():
    assert host_matches("https://cse.iitrpr.ac.in/x", ["iitrpr.ac.in"])
    assert not host_matches("https://iitrpr.ac.in.evil.com/x", ["iitrpr.ac.in"])
    assert host_matches("https://m.facebook.com/p", ["facebook.com"])


def test_query_building():
    assert "site:iitrpr.ac.in OR site:zeitgeist.org.in OR site:advitiya.org.in" in build_query("fest dates", "official")
    assert build_query("Zeitgeist dates", "open") == "Zeitgeist dates IIT Ropar"
    assert build_query("IIT Ropar ranking", "open") == "IIT Ropar ranking"


def test_official_scope_keeps_only_official_domains():
    urls = [r.url for r in searcher().search("zeitgeist", "official")]
    assert urls == ["https://www.zeitgeist.org.in/", "https://hostel.iitrpr.ac.in/rules"]


def test_open_scope_blocks_social_media():
    urls = [r.url for r in searcher().search("zeitgeist", "open")]
    assert "https://www.instagram.com/zeitgeist.iitrpr/" not in urls
    assert "https://someblog.example.com/zeitgeist" in urls


def test_gather_picks_relevant_passage_and_redacts():
    sources = searcher().gather("When is Zeitgeist 2026?", "official")
    zg = next(s for s in sources if "zeitgeist.org.in" in s.url)
    assert zg.source_type == "web" and "30 October to 1 November 2026" in zg.text
    assert "98765" not in zg.text and "[mobile redacted]" in zg.text


# --- engine ---------------------------------------------------------------------------------
@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_WEB_QUERIES_PATH", tmp_path / "web.jsonl")
    builder = StubBuilder()
    builder.tables = TableStore()
    return AnswerEngine(provider=FakeProvider(), fallback=None, builder=builder, store=Store(tmp_path / "s.db"),
                        limiter=RateLimiter(1000), web=searcher())


def run(engine, q, **kw):
    events = list(engine.answer(q, **kw))
    return events, "".join(e["text"] for e in events if e["type"] == "delta")


def web_log():
    return [json.loads(line) for line in config.LOG_WEB_QUERIES_PATH.read_text(encoding="utf-8").splitlines()]


def test_button_triggered_web_answer_is_labelled_logged_and_not_cached(engine):
    events, _ = run(engine, "When is Zeitgeist?", web="official", trigger="thumbs_down")
    meta = events[0]
    assert meta["route"] == "web" and meta["scope"] == "official"
    assert all(s["source_type"] == "web" for s in meta["sources"])
    assert web_log()[-1]["trigger"] == "thumbs_down" and web_log()[-1]["results"] == 2
    events, _ = run(engine, "When is Zeitgeist?", web="official")
    assert not events[0]["cached"]


def test_explicit_request_in_question(engine):
    events, _ = run(engine, "Please search the web for Zeitgeist dates")
    assert events[0]["route"] == "web" and web_log()[-1]["searched"] == "Zeitgeist dates"
    assert engine.web.provider.queries[-1].startswith("Zeitgeist dates (site:")


def test_bare_search_request_uses_previous_question(engine):
    history = [{"role": "user", "content": "When is Zeitgeist?"}, {"role": "assistant", "content": "Not sure."}]
    run(engine, "search the whole web", history=history)
    assert web_log()[-1]["searched"] == "When is Zeitgeist" and web_log()[-1]["scope"] == "open"


@pytest.mark.parametrize("n", [1, 179, 180, 181, 262, 320, 1000])
def test_passages_cover_the_whole_text(n):
    from app.websearch import passages
    words = [f"w{i}" for i in range(n)]
    covered = set(" ".join(passages(" ".join(words))).split())
    assert covered == set(words)


def test_web_search_still_refuses_student_lookups(engine):
    events, text = run(engine, "search the web for the phone number of Rahul from CSE")
    assert events[0]["route"] == "refuse" and engine.web.provider.queries == []


def test_no_results_message_is_logged(engine):
    engine._web = searcher(results=[])
    events, text = run(engine, "When is Zeitgeist?", web="official")
    assert "couldn't find anything" in text and web_log()[-1]["results"] == 0


def test_web_search_never_profiles_private_people(engine):
    events, text = run(engine, "tell all you know about shlok vaidya", web="official", trigger="thumbs_down")
    assert events[0]["route"] == "refuse" and "don't search the web" in text
    assert engine.web.provider.queries == []
