"""The answer pipeline: question -> sources -> LLM stream -> redacted answer, as a stream of events.

Events (plain dicts, sent to the browser as server-sent events):
  {"type": "meta", "route", "reason", "sources": [...], "offer_web", "provider", "model", "cached"}
  {"type": "delta", "text"}
  {"type": "done", "usage", "cost", "cached", "notice"}
  {"type": "error", "code", "message"}
"""
import re
import sys
import time
from collections.abc import Iterator

from app import config
from app.context import Context, ContextBuilder
from app.llm import LLMError, Provider, Usage, get_provider
from app.privacy import StreamRedactor, redact
from app.prompt import build_messages
from app.router import REFUSAL, Route, is_private_person, is_student_lookup, strip_web_phrase, web_request
from app.state import RateLimiter, Store
from app.websearch import WebSearcher, log_web_query

MAX_QUESTION_CHARS = 1000
LIMIT_MESSAGE = ("Ask IIT Ropar has reached its daily usage limit, so I can't answer new questions right now. "
                 "Please try again tomorrow, or check the official site: https://www.iitrpr.ac.in")
PERSON_WEB_REFUSAL = ("I don't search the web for information about individual people. I can only share what IIT Ropar "
                      "officially publishes about its faculty, staff and student office-bearers.")
ERROR_MESSAGE ="Sorry, the answering service is unavailable right now. Please try again in a minute."
# a follow-up like "what about his email?" needs the previous question to retrieve anything useful
FOLLOW_UP = re.compile(r"\b(he|she|him|his|her|they|them|their|it|its|that|this|those|these|there|same|also|what about)\b",
                       re.I)


class AnswerEngine:
    def __init__(self, provider: Provider | None = None, fallback: Provider | None | str = "config",
                 builder: ContextBuilder | None = None, store: Store | None = None,
                 limiter: RateLimiter | None = None, web: WebSearcher | None = None):
        self.provider = provider or get_provider()
        if fallback == "config":
            name = config.BUDGET_FALLBACK_PROVIDER
            fallback = get_provider(name) if name and name != self.provider.name else None
        self.fallback = fallback
        self.builder = builder or ContextBuilder()
        self.store = store or Store()
        self.limiter = limiter or RateLimiter()
        self._web = web

    @property
    def web(self) -> WebSearcher:
        if self._web is None:
            self._web = WebSearcher(self.builder.searcher.model)
        return self._web

    def _web_context(self, question: str, history: list[dict], scope: str) -> tuple[Context, str]:
        """Sources from a web search. Returns (context, the topic that was searched)."""
        topic = strip_web_phrase(question) if web_request(question) else question
        if not topic:  # a bare "search the web" right after an answer means: that question
            topic = next((strip_web_phrase(m["content"]) for m in reversed(history)
                          if m.get("role") == "user" and strip_web_phrase(m["content"])), "")
        if not topic:
            return Context(Route("web", "nothing to search"), [], 0.0, False,
                           "What would you like me to search the web for?"), ""
        if is_student_lookup(topic, self.builder.tables):
            return Context(Route("refuse", "asks for an individual student's details"), [], 1.0, False, REFUSAL), topic
        if is_private_person(topic, self.builder.tables):
            return Context(Route("refuse", "web search about a private individual"), [], 1.0, False,
                           PERSON_WEB_REFUSAL), topic
        sources = self.web.gather(topic, scope)
        route = Route("web", f"web search ({scope})", {"scope": scope})
        if not sources:
            where = "IIT Ropar's official websites" if scope == "official" else "the web"
            return Context(route, [], 0.0, False, f"I couldn't find anything relevant on {where} for that."), topic
        return Context(route, sources, 1.0, False), topic

    @staticmethod
    def retrieval_query(question: str, history: list[dict]) -> str:
        last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
        if last_user and (len(question.split()) <= 6 or FOLLOW_UP.search(question)):
            return f"{last_user} {question}"
        return question

    def _pick_provider(self, messages: list[dict]) -> tuple[Provider | None, str | None]:
        """The provider to use, or None plus a notice when the daily cap is reached and there's no fallback."""
        p = self.provider
        if p.paid:
            spent = self.store.spent_today(p.name)
            if spent + p.worst_case_cost(messages, config.LLM_MAX_TOKENS) > config.DAILY_SPEND_CAP:
                if self.fallback and self.fallback.available():
                    return self.fallback, f"Daily limit for {p.name} reached; answered with {self.fallback.name}."
                return None, LIMIT_MESSAGE
        return p, None

    def answer(self, question: str, history: list[dict] | None = None, client: str = "local",
               web: str | None = None, trigger: str | None = None) -> Iterator[dict]:
        """web='official' | 'open' runs a web search (from the 👎 / "Search the web" buttons); a question that
        explicitly asks to search the web does the same."""
        history = history or []
        question = question.strip()[:MAX_QUESTION_CHARS]
        if not question:
            yield {"type": "error", "code": "empty", "message": "Please type a question."}
            return
        if not self.limiter.allow(client):
            yield {"type": "error", "code": "rate_limited",
                   "message": "You're sending questions too quickly. Please wait a minute and try again."}
            return

        scope = web if web in ("official", "open") else web_request(question)
        topic = None
        if scope:
            ctx, topic = self._web_context(question, history, scope)
            trigger = trigger or "explicit"
        else:
            ctx = self.builder.gather(self.retrieval_query(question, history))
        meta = {"type": "meta", "route": ctx.route.kind, "reason": ctx.route.reason, "offer_web": ctx.offer_web,
                "scope": scope if ctx.route.kind == "web" else None,
                "sources": [{"n": i, "title": s.title, "url": s.url, "source_type": s.source_type}
                            for i, s in enumerate(ctx.sources, 1)],
                "provider": self.provider.name, "model": self.provider.model, "cached": False}

        if ctx.refusal:
            if topic and ctx.route.kind == "web":
                log_web_query(question, topic, scope, trigger, [], self.provider.name)
            yield {**meta, "sources": []}
            yield {"type": "delta", "text": ctx.refusal}
            yield {"type": "done", "usage": None, "cost": 0.0, "cached": False, "notice": None}
            return

        no_cache = history or scope
        cache_key = None if no_cache else self.store.cache_key(question, self.provider.name, self.provider.model)
        if cache_key and (hit := self.store.get_answer(cache_key)):
            text, cached_meta = hit
            yield {**cached_meta, "cached": True}
            yield {"type": "delta", "text": text}
            yield {"type": "done", "usage": None, "cost": 0.0, "cached": True, "notice": None}
            return

        messages = build_messages(question, ctx.sources, history)
        provider, notice = self._pick_provider(messages)
        if provider is None:
            yield meta
            yield {"type": "delta", "text": notice}
            yield {"type": "done", "usage": None, "cost": 0.0, "cached": False, "notice": "limit_reached"}
            return

        yield {**meta, "provider": provider.name, "model": provider.model}
        parts, usage, redactor = [], None, StreamRedactor()
        used = [provider]  # _generate swaps in the fallback if the paid provider is out of credit
        try:
            for ev in self._generate(used, messages):
                if isinstance(ev, Usage):
                    usage = ev
                elif safe := redactor.feed(ev):
                    parts.append(safe)
                    yield {"type": "delta", "text": safe}
        except LLMError as e:
            print(f"[llm] {provider.name} failed: {e}", file=sys.stderr)
            if tail := redactor.flush():
                yield {"type": "delta", "text": tail}
            yield {"type": "error", "code": "llm_error", "message": ERROR_MESSAGE}
            return
        if tail := redactor.flush():
            parts.append(tail)
            yield {"type": "delta", "text": tail}

        if used[0] is not provider:
            notice = f"{provider.name} is out of credit; answered with {used[0].name}."
        provider = used[0]
        if usage:
            self.store.record_usage(provider.name, provider.model, usage)
        if ctx.route.kind == "web":
            log_web_query(question, topic, scope, trigger, ctx.sources, provider.name)
        text = redact("".join(parts))
        if cache_key and provider is self.provider and text.strip():
            ttl = config.LIVE_CACHE_TTL_MINUTES * 60 if ctx.route.kind == "live" else config.ANSWER_CACHE_TTL_HOURS * 3600
            self.store.put_answer(cache_key, question, text, {**meta, "provider": provider.name}, ttl)
        yield {"type": "done", "cached": False, "notice": notice, "cost": usage.cost if usage else 0.0,
               "usage": usage and {"prompt_tokens": usage.prompt_tokens, "cached_tokens": usage.cached_tokens,
                                   "completion_tokens": usage.completion_tokens, "estimated": usage.estimated}}

    def _generate(self, used: list[Provider], messages: list[dict]):
        """Stream from used[0]; if a paid provider is out of credit before any text, switch to the fallback."""
        started = False
        try:
            for ev in used[0].generate(messages, stream=True):
                started = started or isinstance(ev, str)
                yield ev
        except LLMError as e:
            if started or not e.out_of_credit or not self.fallback or not self.fallback.available():
                raise
            print(f"[llm] {used[0].name} out of credit; falling back to {self.fallback.name}", file=sys.stderr)
            used[0] = self.fallback
            yield from self.fallback.generate(messages, stream=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ask a question from the command line.")
    ap.add_argument("questions", nargs="+")
    ap.add_argument("--provider", help="pollinations | ollama | fake (default: LLM_PROVIDER)")
    ap.add_argument("--web", choices=["official", "open"], help="answer from a web search instead")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    engine = AnswerEngine(provider=get_provider(args.provider))
    for q in args.questions:
        print(f"\n{'=' * 100}\nQ: {q}")
        t0, first = time.time(), None
        for ev in engine.answer(q, web=args.web, trigger="cli" if args.web else None):
            if ev["type"] == "meta":
                print(f"[route={ev['route']} scope={ev['scope']} provider={ev['provider']}:{ev['model']} "
                      f"sources={len(ev['sources'])} offer_web={ev['offer_web']} cached={ev['cached']}]")
                for s in ev["sources"]:
                    print(f"  [{s['n']}] {s['source_type']:4} {s['title'][:70]} | {s['url']}")
                print()
            elif ev["type"] == "delta":
                first = first or time.time()
                print(ev["text"], end="", flush=True)
            elif ev["type"] == "done":
                u = ev["usage"] or {}
                print(f"\n\n[done in {time.time() - t0:.1f}s, first text after {(first or time.time()) - t0:.1f}s; "
                      f"tokens in={u.get('prompt_tokens')} out={u.get('completion_tokens')} cost={ev['cost']:.6f} "
                      f"cached={ev['cached']} notice={ev['notice']}]")
            elif ev["type"] == "error":
                print(f"\n[error: {ev['code']}] {ev['message']}")
