"""Answer pipeline tests. They use the fake provider only: no network, no Ollama, no paid API."""
import pytest

from app import config
from app.answer import LIMIT_MESSAGE, AnswerEngine
from app.context import Context
from app.llm.base import LLMError, Provider, Usage
from app.llm.fake import FakeProvider
from app.privacy import StreamRedactor
from app.prompt import build_messages, system_prompt
from app.router import Route
from app.sources import Source
from app.state import RateLimiter, Store

SOURCES = [Source("Hostel Rules AY 2025-26", "https://www.iitrpr.ac.in/hostel-rules.pdf", "pdf", "2026-10-22",
                  "Visitors are not allowed in hostel rooms after 10 pm.")]


class StubBuilder:
    """Stands in for ContextBuilder so these tests don't load the index."""

    def __init__(self, sources=SOURCES, kind="retrieval", refusal=None):
        self.sources, self.kind, self.refusal = sources, kind, refusal
        self.queries = []

    def gather(self, q):
        self.queries.append(q)
        return Context(Route(self.kind, "test"), self.sources, 0.8, False, self.refusal)


class PaidFake(FakeProvider):
    name, paid = "paidfake", True

    def __init__(self, cost_each=0.4, **kw):
        super().__init__(**kw)
        self.cost_each, self.calls = cost_each, 0

    def worst_case_cost(self, messages, max_tokens):
        return self.cost_each

    def generate(self, messages, stream=True):
        self.calls += 1
        for ev in super().generate(messages, stream):
            yield Usage(ev.prompt_tokens, ev.completion_tokens, cost=self.cost_each) if isinstance(ev, Usage) else ev


def run(engine, q="What are the hostel rules?", **kw):
    events = list(engine.answer(q, **kw))
    text = "".join(e["text"] for e in events if e["type"] == "delta")
    return events, text


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "state.db")


def engine_with(store, provider=None, fallback=None, **builder_kw):
    return AnswerEngine(provider=provider or FakeProvider(), fallback=fallback, builder=StubBuilder(**builder_kw),
                        store=store, limiter=RateLimiter(1000))


# --- prompt ----------------------------------------------------------------------------
def test_system_prompt_has_date_and_is_stable():
    a, b = system_prompt("Thursday, 22 October 2026"), system_prompt("Thursday, 22 October 2026")
    assert a == b and "22 October 2026" in a and "{today}" not in a


def test_user_message_carries_numbered_sources_and_history_has_no_old_context():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    msgs = build_messages("What are the hostel rules?", SOURCES, history)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"].startswith("CONTEXT:\n[1] Hostel Rules AY 2025-26 | https://www.iitrpr.ac.in/")
    assert "source: pdf | fetched: 2026-10-22" in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith("QUESTION: What are the hostel rules?")


# --- streaming redaction ------------------------------------------------------------------
@pytest.mark.parametrize("secret", ["2023CSB1234", "98765 43210", "+91-98765-43210", "9876543210"])
@pytest.mark.parametrize("piece", [1, 3, 7, 50])
def test_redaction_survives_any_stream_split(secret, piece):
    text = f"Contact the student at {secret} for details. Office: 01881-23-2652."
    r, out = StreamRedactor(), []
    for i in range(0, len(text), piece):
        out.append(r.feed(text[i:i + piece]))
    out.append(r.flush())
    joined = "".join(out)
    assert secret not in joined and "redacted]" in joined
    assert "01881-23-2652" in joined  # official office numbers are kept


def test_engine_redacts_streamed_answer(store):
    eng = engine_with(store, provider=FakeProvider(script="Call 98765 43210 or 2024MEB5678 today.", piece=2))
    _, text = run(eng)
    assert "98765" not in text and "2024MEB5678" not in text


# --- pipeline -----------------------------------------------------------------------------
def test_answer_event_order_and_sources(store):
    events, text = run(engine_with(store))
    assert [e["type"] for e in events][0] == "meta" and events[-1]["type"] == "done"
    assert events[0]["sources"][0]["url"].endswith("hostel-rules.pdf")
    assert "[1]" in text


def test_refusal_skips_llm(store):
    provider = PaidFake()
    events, text = run(engine_with(store, provider=provider, refusal="No student lookups."))
    assert text == "No student lookups." and provider.calls == 0


def test_cache_hit_on_repeat_question(store):
    provider = PaidFake(cost_each=0.01)
    eng = engine_with(store, provider=provider)
    _, first = run(eng, "What are the hostel rules?")
    events, second = run(eng, "what are the HOSTEL rules")
    assert second == first and events[-1]["cached"] and provider.calls == 1


def test_no_cache_with_history(store):
    provider = PaidFake(cost_each=0.01)
    eng = engine_with(store, provider=provider)
    run(eng)
    run(eng, history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert provider.calls == 2


def test_daily_cap_falls_back_then_shows_limit(store, monkeypatch):
    monkeypatch.setattr(config, "DAILY_SPEND_CAP", 1.0)
    paid = PaidFake(cost_each=0.4)
    eng = engine_with(store, provider=paid, fallback=FakeProvider(script="free answer"))
    run(eng, "q1"), run(eng, "q2")                      # spends 0.8
    events, text = run(eng, "q3")                       # 0.8 + 0.4 > 1.0 -> fallback
    assert text == "free answer" and paid.calls == 2 and "Daily limit" in events[-1]["notice"]

    eng.fallback = None
    events, text = run(eng, "q4")
    assert text == LIMIT_MESSAGE and events[-1]["notice"] == "limit_reached" and paid.calls == 2


def test_out_of_credit_error_uses_fallback(store):
    class Broke(Provider):
        name, paid = "broke", True

        def generate(self, messages, stream=True):
            raise LLMError("HTTP 402", 402, out_of_credit=True)
            yield  # pragma: no cover

    eng = engine_with(store, provider=Broke("x"), fallback=FakeProvider(script="fallback answer"))
    events, text = run(eng)
    assert text == "fallback answer" and "out of credit" in events[-1]["notice"]


def test_rate_limit(store):
    eng = AnswerEngine(provider=FakeProvider(), fallback=None, builder=StubBuilder(), store=store,
                       limiter=RateLimiter(2))
    assert run(eng, "a", client="1.2.3.4")[0][-1]["type"] == "done"
    assert run(eng, "b", client="1.2.3.4")[0][-1]["type"] == "done"
    assert run(eng, "c", client="1.2.3.4")[0][-1]["code"] == "rate_limited"
    assert run(eng, "c", client="5.6.7.8")[0][-1]["type"] == "done"


def test_follow_up_uses_previous_question_for_retrieval(store):
    eng = engine_with(store)
    history = [{"role": "user", "content": "Who is the BoCA faculty advisor?"},
               {"role": "assistant", "content": "Dr. Anupam Bandyopadhyay [1]"}]
    run(eng, "what is his email?", history=history)
    assert eng.builder.queries[-1].startswith("Who is the BoCA faculty advisor?")
