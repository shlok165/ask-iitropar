"""One command to run Ask IIT Ropar locally: python -m app

Builds the search index on first run (~10 min on CPU), then serves the chat page.
"""
import chromadb
import uvicorn

from app import config
from app.index import COLLECTION, build


def index_ready() -> bool:
    try:
        return chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(COLLECTION).count() > 0
    except Exception:
        return False


if __name__ == "__main__":
    if not index_ready():
        print("No search index yet; building it first (one-time, about 10 minutes on CPU)...")
        build()
    print(f"Ask IIT Ropar: http://{'localhost' if config.HOST in ('127.0.0.1', '0.0.0.0') else config.HOST}:{config.PORT}")
    uvicorn.run("app.server:app", host=config.HOST, port=config.PORT, workers=1)
