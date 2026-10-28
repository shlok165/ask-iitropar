"""Hybrid search: embeddings (Chroma) + BM25 keywords, merged with weighted reciprocal rank fusion."""
import re
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app import config
from app.index import COLLECTION, load_chunks

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
RRF_K = 60
POOL = 40  # candidates taken from each ranker before merging

# Alphanumeric runs stay whole, so course codes ("cs201") and acronyms ("boca") match exactly.
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the of in on at to for is are was were be been and or with by from as it its this that "
                  "what who when where which how do does i me my you your can about".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    chunk_id: str
    title: str
    url: str
    category: str
    source_type: str
    fetched: str
    text: str
    score: float                  # fused score (higher is better)
    cosine: float | None = None   # similarity to the query, only if the dense ranker returned it
    dense_rank: int | None = None
    bm25_rank: int | None = None


class HybridSearcher:
    def __init__(self):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
        self.col = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(COLLECTION)
        self.chunks = load_chunks()
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.by_id = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}

    def _dense(self, query: str, n: int, category: str | None):
        emb = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
        r = self.col.query(query_embeddings=emb, n_results=n, where={"category": category} if category else None,
                           include=["documents", "metadatas", "distances"])
        return [(i, d, m, 1.0 - dist) for i, d, m, dist in
                zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def _bm25(self, query: str, n: int, category: str | None):
        scores = self.bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1]
        out = []
        for idx in order:
            if scores[idx] <= 0 or len(out) >= n:
                break
            c = self.chunks[idx]
            if category and c["category"] != category:
                continue
            out.append(c["chunk_id"])
        return out

    def search(self, query: str, k: int | None = None, alpha: float | None = None,
               category: str | None = None) -> list[Hit]:
        k = k or config.TOP_K
        alpha = config.HYBRID_ALPHA if alpha is None else alpha
        dense = self._dense(query, POOL, category)
        sparse = self._bm25(query, POOL, category) if alpha < 1 else []

        fused: dict[str, Hit] = {}
        for rank, (cid, doc, meta, cos) in enumerate(dense, 1):
            fused[cid] = Hit(cid, meta["title"], meta["url"], meta["category"], meta["source_type"],
                             meta["fetched"], doc, alpha / (RRF_K + rank), cosine=cos, dense_rank=rank)
        for rank, cid in enumerate(sparse, 1):
            w = (1 - alpha) / (RRF_K + rank)
            if cid in fused:
                fused[cid].score += w
                fused[cid].bm25_rank = rank
            else:
                c = self.chunks[self.by_id[cid]]
                fused[cid] = Hit(cid, c["title"], c["url"], c["category"], c["source_type"],
                                 c["fetched"], c["text"], w, bm25_rank=rank)
        return sorted(fused.values(), key=lambda h: -h.score)[:k]

    def confidence(self, hits: list[Hit]) -> float:
        """Best cosine similarity among the returned hits; below MIN_RELEVANCE we offer a web search."""
        return max((h.cosine for h in hits if h.cosine is not None), default=0.0)


if __name__ == "__main__":
    import sys
    s = HybridSearcher()
    for q in sys.argv[1:] or ["Who teaches machine learning in CSE?"]:
        hits = s.search(q)
        print(f"\nQ: {q}   (confidence {s.confidence(hits):.2f})")
        for h in hits:
            print(f"  {h.score:.4f} d={h.dense_rank} b={h.bm25_rank} [{h.category}/{h.source_type}] {h.title[:60]} | {h.url}")
