"""Gathers the sources for a question: route it, run that path, fall back to hybrid retrieval if it comes back empty."""
from dataclasses import dataclass

from app import config
from app.live import LiveAPI
from app.router import REFUSAL, Route, is_private_person, route
from app.search import Hit, HybridSearcher
from app.sources import Source, format_sources
from app.tables import TableStore

SUPPLEMENT_K = 3  # retrieval chunks added after table rows (e.g. course pages for "who teaches X")


@dataclass
class Context:
    route: Route
    sources: list[Source]
    confidence: float      # 1.0 for table/live answers; best cosine similarity for retrieval
    offer_web: bool        # low confidence -> offer (not run) a web search
    refusal: str | None = None


def hits_to_sources(hits: list[Hit]) -> list[Source]:
    """One source per page: chunks from the same URL are joined, so citations and source cards aren't duplicated."""
    by_url: dict[str, Source] = {}
    for h in hits:
        if h.url in by_url:
            by_url[h.url].text += "\n[...]\n" + h.text
        else:
            by_url[h.url] = Source(h.title, h.url, h.source_type, h.fetched, h.text)
    return list(by_url.values())


class ContextBuilder:
    def __init__(self, searcher: HybridSearcher | None = None, tables: TableStore | None = None,
                 live: LiveAPI | None = None):
        self.searcher = searcher or HybridSearcher()
        self.tables = tables or TableStore()
        self.live = live or LiveAPI()

    def _retrieve(self, question: str, k: int | None = None) -> tuple[list[Source], float]:
        hits = self.searcher.search(question, k=k)
        return hits_to_sources(hits), self.searcher.confidence(hits)

    def gather(self, question: str) -> Context:
        r = route(question, self.tables)
        if r.kind == "refuse":
            return Context(r, [], 1.0, False, REFUSAL)

        sources: list[Source] = []
        if r.kind == "table":
            sources = self.tables.lookup(question)
            if sources and "people" not in r.entities:  # a named person's row needs no supporting pages
                extra, _ = self._retrieve(question, k=SUPPLEMENT_K)
                seen = {s.url for s in sources} | {s.title for s in sources}
                sources += [s for s in extra if s.url not in seen and s.title not in seen]
        elif r.kind == "live":
            sources = self.live.search(question)

        if sources:
            return Context(r, sources, 1.0, False)
        if r.kind != "retrieval":
            r = Route("retrieval", f"{r.kind} path found nothing ({r.reason})", r.entities)
        sources, conf = self._retrieve(question)
        # never suggest a web search for a profile of a private individual (possibly a student)
        offer_web = conf < config.MIN_RELEVANCE and not is_private_person(question, self.tables)
        return Context(r, sources, conf, offer_web)


if __name__ == "__main__":
    import sys
    builder = ContextBuilder()
    verbose = "-v" in sys.argv
    for q in [a for a in sys.argv[1:] if a != "-v"]:
        ctx = builder.gather(q)
        print(f"\n{'=' * 100}\nQ: {q}\nRoute: {ctx.route.kind} ({ctx.route.reason}) {ctx.route.entities or ''}"
              f"\nConfidence: {ctx.confidence:.2f}  offer web search: {ctx.offer_web}")
        if ctx.refusal:
            print("Refusal:", ctx.refusal)
        if verbose:
            print(format_sources(ctx.sources)[:3000])
        else:
            for i, s in enumerate(ctx.sources, 1):
                print(f"  [{i}] {s.source_type:4} {s.title[:75]} | {s.url}")
