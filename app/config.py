"""Settings: real environment variables win over .env, which wins over the defaults below."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.split(" #", 1)[0].strip().strip('"').strip("'")
    return env


_DOTENV = _load_dotenv(ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name) or _DOTENV.get(name) or default


def _path(name: str, default: str) -> Path:
    p = Path(_get(name, default))
    return p if p.is_absolute() else ROOT / p


# --- Retrieval -----------------------------------------------------------------
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_DIR = _path("CHROMA_DIR", "index/chroma")
CHUNKS_PATH = _path("CHUNKS_PATH", "data/chunks.jsonl")
CORPUS_PATH = _path("CORPUS_PATH", "data/corpus.jsonl")
TOP_K = int(_get("TOP_K", "6"))
HYBRID_ALPHA = float(_get("HYBRID_ALPHA", "0.5"))  # 0 = BM25 only, 1 = embeddings only
# Best cosine similarity below this -> offer a web search. Calibrated on eval/questions.jsonl: answerable
# questions scored >= 0.68, out-of-scope ones 0.57-0.80 (bge-small scores run high), see eval/report.md.
MIN_RELEVANCE = float(_get("MIN_RELEVANCE", "0.65"))

# --- Live IIT Ropar API --------------------------------------------------------------
IITRPR_API_BASE = _get("IITRPR_API_BASE", "https://www.iitrpr.ac.in/api/")

# --- LLM providers ---------------------------------------------------------------------
LLM_PROVIDER = _get("LLM_PROVIDER", "pollinations")  # pollinations | ollama | fake
LLM_REASONING_EFFORT = _get("LLM_REASONING_EFFORT", "low")
LLM_MAX_TOKENS = int(_get("LLM_MAX_TOKENS", "800"))  # hidden reasoning tokens count toward this too

POLLINATIONS_API_KEY = _get("POLLINATIONS_API_KEY")
POLLINATIONS_BASE_URL = _get("POLLINATIONS_BASE_URL", "https://gen.pollinations.ai/v1")
POLLINATIONS_MODEL = _get("POLLINATIONS_MODEL", "openai/gpt-5.4-mini")

OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_NUM_CTX = int(_get("OLLAMA_NUM_CTX", "8192"))  # Ollama's default (4096) would cut our prompts


# --- Budget, cache, limits -------------------------------------------------------------
DAILY_SPEND_CAP = float(_get("DAILY_SPEND_CAP", "1.0"))  # per paid provider per day (pollen, or USD)
BUDGET_FALLBACK_PROVIDER = _get("BUDGET_FALLBACK_PROVIDER", "ollama")  # used when over cap; "" = show a message
ANSWER_CACHE_TTL_HOURS = float(_get("ANSWER_CACHE_TTL_HOURS", "24"))
LIVE_CACHE_TTL_MINUTES = float(_get("LIVE_CACHE_TTL_MINUTES", "20"))  # cached answers built from live data
HISTORY_TURNS = int(_get("HISTORY_TURNS", "3"))
STATE_DB_PATH = _path("STATE_DB_PATH", "logs/state.db")
RATE_LIMIT_PER_MINUTE = int(_get("RATE_LIMIT_PER_MINUTE", "20"))
TIMEZONE = _get("TIMEZONE", "Asia/Kolkata")

# --- Web-search fallback --------------------------------------------------------------------
WEB_SEARCH_PROVIDER = _get("WEB_SEARCH_PROVIDER", "ddgs")
WEB_SEARCH_MAX_RESULTS = int(_get("WEB_SEARCH_MAX_RESULTS", "5"))  # pages fetched per web answer
WEB_ALLOWED_DOMAINS = [d.strip() for d in _get(
    "WEB_ALLOWED_DOMAINS", "iitrpr.ac.in,zeitgeist.org.in,advitiya.org.in").split(",") if d.strip()]
WEB_BLOCKED_DOMAINS = [d.strip() for d in _get(
    "WEB_BLOCKED_DOMAINS", "facebook.com,instagram.com,x.com,twitter.com,linkedin.com,youtube.com,reddit.com,"
                           "quora.com,tiktok.com,snapchat.com,threads.net,pinterest.com,t.me,telegram.me,"
                           "whatsapp.com,discord.com,discord.gg").split(",") if d.strip()]
LOG_WEB_QUERIES_PATH = _path("LOG_WEB_QUERIES_PATH", "logs/web_queries.jsonl")

# --- App -----------------------------------------------------------------------------------
HOST = _get("HOST", "127.0.0.1")  # the Docker image sets 0.0.0.0
PORT = int(_get("PORT", "7860"))   # Hugging Face Spaces expects 7860
FEEDBACK_LOG_PATH = _path("FEEDBACK_LOG_PATH", "logs/feedback.jsonl")
