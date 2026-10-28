"""Build the vector index: python -m app.index

Embeds data/chunks.jsonl with bge-small into a persisted ChromaDB collection.
BM25 is built in memory at search time (see app/search.py), so nothing else to build.
"""
import json
import time

import chromadb
from sentence_transformers import SentenceTransformer

from app import config

COLLECTION = "iitrpr_chunks"
BATCH = 64


def load_chunks() -> list[dict]:
    """Chunks plus the fetched date of the document each came from."""
    fetched = {}
    with config.CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fetched[d["id"]] = (d.get("fetched_at") or "")[:10]
    chunks, seen = [], {}
    with config.CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            # The scraper gives some different pages the same doc id (same url+title); keep both.
            n = seen[c["chunk_id"]] = seen.get(c["chunk_id"], -1) + 1
            if n:
                c["chunk_id"] += f"~{n}"
            c["fetched"] = fetched.get(c["doc_id"], "")
            # official source type for the prompt: site | pdf | api
            c["source_type"] = c["source"].split(":", 1)[0]
            chunks.append(c)
    return chunks


def build() -> None:
    chunks = load_chunks()
    print(f"{len(chunks)} chunks; loading {config.EMBEDDING_MODEL} ...", flush=True)
    model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    t0 = time.time()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        emb = model.encode([c["text"] for c in batch], batch_size=BATCH, normalize_embeddings=True)
        col.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=emb.tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[{"url": c["url"], "title": c["title"], "category": c["category"],
                        "source_type": c["source_type"], "fetched": c["fetched"]} for c in batch],
        )
        done = i + len(batch)
        if (i // BATCH) % 10 == 0 or done == len(chunks):
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(chunks)}  ({rate:.0f} chunks/s, ~{(len(chunks) - done) / rate:.0f}s left)", flush=True)
    print(f"Done: {col.count()} chunks indexed in {time.time() - t0:.0f}s -> {config.CHROMA_DIR}")


if __name__ == "__main__":
    build()
