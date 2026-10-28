"""Shared helpers: paths, HTTP session, text cleaning, privacy redaction, doc storage."""
import hashlib
import json
import re
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urlparse

import requests
import urllib3

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DOCS = DATA / "docs"
for d in (RAW, DOCS):
    d.mkdir(parents=True, exist_ok=True)

BASE = "https://www.iitrpr.ac.in"
API = BASE + "/api/"
UA = "Mozilla/5.0 (compatible; IITRoparRAGBot/1.0; student knowledge-base project)"

# --- Privacy -----------------------------------------------------------------
# Pages/files that list individual students or applicants. We never collect them.
EXCLUDE_URL_RE = re.compile(
    r"(on-roll-students"
    r"|admission-file/[^?]*(select|slected|shortlist|provisional|result|candidate|merit|interview|SL_)"
    r"|/admin|/login|studentportal|/signup|/callback|/forbidden|logout"
    r"|tender-file/|/v2/files/)",
    re.I,
)
# Student entry numbers (e.g. 2023MEB1326) and Indian mobile numbers.
ENTRY_NO_RE = re.compile(r"\b20\d{2}[A-Z]{3}\d{4}\b")
MOBILE_RE = re.compile(r"(?<![\d-])(?:\+?91[\s-]*)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")


def redact(text: str) -> str:
    text = ENTRY_NO_RE.sub("[entry-no redacted]", text)
    return MOBILE_RE.sub("[mobile redacted]", text)


def is_excluded(url: str) -> bool:
    return bool(EXCLUDE_URL_RE.search(url))


# --- HTTP ----------------------------------------------------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_session = requests.Session()
_session.headers["User-Agent"] = UA
_robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def robots_allowed(url: str) -> bool:
    p = urlparse(url)
    key = f"{p.scheme}://{p.netloc}"
    if key not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = _session.get(key + "/robots.txt", timeout=10, verify=False)
            ok = r.status_code == 200 and "text/plain" in r.headers.get("content-type", "")
            rp.parse(r.text.splitlines()) if ok else None
            _robots[key] = rp if ok else None
        except requests.RequestException:
            _robots[key] = None
    rp = _robots[key]
    return rp is None or rp.can_fetch(UA, url)


def get(url: str, timeout=30, stream=False, delay=0.3) -> requests.Response | None:
    time.sleep(delay)
    for verify in (True, False):
        try:
            return _session.get(url, timeout=timeout, stream=stream, verify=verify)
        except requests.exceptions.SSLError:
            continue
        except requests.RequestException as e:
            print(f"  ! {url}: {e.__class__.__name__}")
            return None
    return None


def get_json(path: str):
    r = get(API + path)
    if r is None or r.status_code != 200:
        return None
    try:
        return r.json()
    except ValueError:
        return None


# --- Text & URLs -----------------------------------------------------------------
def normalize_url(url: str) -> str:
    """Drop fragments, except hash-router routes like '#/home'."""
    base, frag = urldefrag(url)
    if frag.startswith("/"):
        return base + "#" + frag
    return base.rstrip("/") if urlparse(base).path not in ("", "/") else base


def clean_text(text: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    out, blank = [], 0
    for ln in lines:
        if not ln:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(ln)
    return "\n".join(out).strip()


CATEGORY_RULES = [
    ("fests", r"zeitgeist|advitiya|aarohan|fest"),
    ("student-life", r"board|club|council|student-affairs|gymkhana|bosa|bost|boca|bola|boha|boaa|ismp|enactus|society"),
    ("hostels", r"hostel|mess|guest-house"),
    ("placements", r"placement|cdpc|caps\.|internship|career"),
    ("admissions", r"admission|jee|phd|/ug-|/pg-|fee-structure|scholarship|fellowship|pmrf"),
    ("academics", r"academic|curriculum|ordinance|course|acug|acpgr|calendar|holiday|handbook|convocation|library"),
    ("faculty", r"faculty|professor"),
    ("research", r"research|centre|center|coe|crf|lab|rnd|tbif|drif|ihub|annam|creed|cdas|datascience"),
    ("departments", r"department|/dept|cse\.|ee\.|mech|civil|chem|phy|math|hss|saide|dbme|mme"),
    ("administration", r"director|dean|senate|governor|committee|statute|rti|tender|job|position|office|organisation|staff|directory"),
    ("facilities", r"medical|bus|transport|it-|/it$|workshop|hpc|day-care|creche|campus"),
    ("news-events", r"news|event|notification|circular|newsletter"),
]


def categorize(url: str, title: str = "") -> str:
    hay = f"{url} {title}".lower()
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, hay):
            return cat
    return "general"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_doc(url: str, title: str, text: str, source: str, category: str | None = None, **meta) -> dict:
    text = clean_text(text)
    return {
        "id": hashlib.sha1(f"{url}|{title}".encode()).hexdigest()[:16],
        "url": url,
        "title": clean_text(title) or url,
        "category": category or categorize(url, title),
        "source": source,
        "text": text,
        "fetched_at": now_iso(),
        **meta,
    }


def save_docs(name: str, docs: list[dict]) -> Path:
    path = DOCS / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[saved] {len(docs)} docs -> {path.relative_to(ROOT)}")
    return path


def save_raw_json(name: str, obj) -> None:
    (RAW / "api").mkdir(exist_ok=True)
    (RAW / "api" / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
