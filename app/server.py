"""FastAPI app: streaming POST /chat (SSE), POST /feedback, GET /health, and the chat page at /."""
import json
import mimetypes
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config
from app.answer import ERROR_MESSAGE, AnswerEngine
from app.privacy import redact

STATIC = config.ROOT / "static"
# Windows can map .js to text/plain in its registry; with nosniff the browser would then refuse the scripts.
for _type, _ext in [("text/javascript", ".js"), ("text/css", ".css"), ("image/svg+xml", ".svg")]:
    mimetypes.add_type(_type, _ext)
engine: AnswerEngine | None = None
_feedback_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine
    engine = AnswerEngine()  # loads the embedding model, index, tables once
    print(f"[ask-iitrpr] ready: provider={engine.provider.name} model={engine.provider.model}", flush=True)
    yield


app = FastAPI(title="Ask IIT Ropar", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'self'")
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    web: Literal["official", "open"] | None = None   # set by the 👎 / "Search the web" buttons
    trigger: Literal["thumbs_down", "offer", "wider"] | None = None


class Feedback(BaseModel):
    rating: Literal["up", "down"]
    question: str = Field(max_length=1000)
    answer: str = Field(default="", max_length=8000)
    route: str | None = Field(default=None, max_length=20)
    sources: list[str] = Field(default_factory=list, max_length=30)


def client_ip(request: Request) -> str:
    # Behind one proxy (Hugging Face), the last X-Forwarded-For entry is the address that proxy saw;
    # earlier entries can be set by the client, so they aren't trusted for rate limiting.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    events = engine.answer(req.question, [t.model_dump() for t in req.history], client=client_ip(request),
                           web=req.web, trigger=req.trigger)

    def sse():
        try:
            for ev in events:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception:  # never leave the browser hanging on an unexpected failure
            traceback.print_exc(file=sys.stderr)
            yield f"data: {json.dumps({'type': 'error', 'code': 'server', 'message': ERROR_MESSAGE})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/feedback")
def feedback(fb: Feedback):
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "rating": fb.rating, "route": fb.route,
              "question": redact(fb.question), "answer": redact(fb.answer)[:2000], "sources": fb.sources}
    config.FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _feedback_lock, config.FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok", "provider": engine.provider.name, "model": engine.provider.model,
            "chunks": engine.builder.searcher.col.count()}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
