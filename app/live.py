"""Live data from the public IIT Ropar JSON API, for time-sensitive questions."""
import re
import time
from datetime import date, timedelta
from urllib.parse import quote

import requests

from app import config
from app.search import tokenize
from app.sources import Source

BASE = "https://www.iitrpr.ac.in"
CACHE_SECONDS = 20 * 60
MAX_ITEMS = 8

FEEDS = {
    "news": ("get-public-global-news", "news"),
    "admissions": ("admissions/active?status=1", "admissions"),
    "jobs": ("v2/public/advertisements?status=1", "advertisements"),
    "calendar": (None, "events"),  # path depends on the date window
}
FEED_WORDS = {
    "admissions": r"admission|admissions|phd|ph\.d|m\.?tech|msc|m\.sc|pg|apply|application|deadline|last date|intake",
    "jobs": r"job|jobs|vacanc|recruit|position|positions|opening|openings|hiring|project staff|jrf|srf|post\b",
    "calendar": r"holiday|holidays|vacation|calendar|break|festival|off day|day off",
    "news": r"news|notice|notices|announcement|announcements|circular|event|events|update|updates|happening",
}
# words that pick the feed rather than filter items inside it
GENERIC = set("latest recent new upcoming current currently today week month next notice notices news update updates "
              "announcement announcements circular event events admission admissions deadline deadlines last date "
              "dates open opening openings ongoing iit ropar iitrpr holiday holidays list any there job jobs vacancy "
              "vacancies recruitment hiring position positions post posts apply application applications still "
              "accepting".split())


def _fetch(path: str):
    r = requests.get(config.IITRPR_API_BASE + path, timeout=15, headers={"User-Agent": "AskIITRopar/1.0"})
    r.raise_for_status()
    return r.json()


class LiveAPI:
    def __init__(self, fetch=_fetch):
        self.fetch = fetch
        self._cache: dict[str, tuple[float, list]] = {}

    def items(self, feed: str) -> list[dict]:
        path, key = FEEDS[feed]
        if feed == "calendar":
            start = date.today() - timedelta(days=30)
            path = f"calendar-events?from={start}&to={start + timedelta(days=364)}"  # API allows ~1 year per call
        hit = self._cache.get(path)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]
        data = self.fetch(path) or {}
        items = data.get(key) or []
        self._cache[path] = (time.time(), items)
        return items

    def search(self, question: str) -> list[Source]:
        ql = question.lower()
        feeds = [f for f, pat in FEED_WORDS.items() if re.search(pat, ql)] or ["news", "admissions"]
        terms = [t for t in tokenize(question) if t not in GENERIC]
        today = date.today().isoformat()
        sources = []
        for feed in feeds:
            try:
                items = self.items(feed)
            except (requests.RequestException, ValueError):
                continue
            sources += getattr(self, f"_{feed}")(items, terms, today)
        return sources

    # --- per-feed formatting --------------------------------------------------------
    @staticmethod
    def _filter(items: list[dict], terms: list[str], text_of) -> list[dict]:
        """Items mentioning the most of the question's specific words (e.g. 'phd', 'cse').
        No specific words -> everything; none match -> nothing (that feed isn't about this question)."""
        if not terms:
            return items
        scored = [(sum(t in text_of(it).lower() for t in terms), it) for it in items]
        best = max((s for s, _ in scored), default=0)
        return [it for s, it in scored if s == best] if best else []

    def _news(self, items, terms, today):
        items = self._filter(items, terms, lambda n: f"{n.get('title')} {n.get('dept_code')} {n.get('type')}")
        items = sorted(items, key=lambda n: n.get("created_at") or "", reverse=True)[:MAX_ITEMS]
        out = []
        for n in items:
            kind = n.get("type") or "news"
            url = f"{BASE}/event/{n['news_id']}" if kind == "event" else f"{BASE}/news/{n['news_id']}"
            if n.get("page_link"):
                url = BASE + n["page_link"] if n["page_link"].startswith("/") else n["page_link"]
            lines = [f"{kind.title()}: {n.get('title')}", f"Posted: {(n.get('created_at') or '')[:10]}"]
            for k, label in [("event_date", "Event date"), ("last_date", "Last date"), ("event_location", "Venue")]:
                if n.get(k):
                    lines.append(f"{label}: {str(n[k])[:10]}")
            if n.get("dept_code"):
                lines.append(f"Department: {n['dept_code'].upper()}")
            out.append(Source(f"{kind.title()}: {n.get('title')}", url, "live", today, "\n".join(lines)))
        return out

    def _admissions(self, items, terms, today):
        items = self._filter(items, terms, lambda a: f"{a.get('title')} {a.get('type')} {a.get('department')} "
                                                     f"{a.get('academic_year')}")
        # open ones first (latest deadline first), then closed ones
        items = sorted(items, key=lambda a: ((a.get("last_date") or "9999")[:10] >= today,
                                             a.get("created_at") or ""), reverse=True)[:MAX_ITEMS]
        out = []
        for a in items:
            last = (a.get("last_date") or "")[:10]
            status = "" if not last else (" (open)" if last >= today else " (closed)")
            lines = [f"Admission notice: {a.get('title')}", f"Posted: {(a.get('created_at') or '')[:10]}"]
            if last:
                lines.append(f"Last date to apply: {last}{status}")
            for k, label in [("type", "Programme type"), ("department", "Department"), ("academic_year", "Academic year"),
                             ("application_link", "Apply at")]:
                if a.get(k):
                    lines.append(f"{label}: {a[k].upper() if k == 'department' else a[k]}")
            for f in a.get("files") or []:
                if f.get("download_url"):
                    lines.append(f"Document: {f.get('name')} {config.IITRPR_API_BASE}{quote(f['download_url'])}")
            out.append(Source(f"Admission: {a.get('title')}", f"{BASE}/admission-notifications#{a.get('id')}", "live",
                              today, "\n".join(lines)))
        return out

    def _jobs(self, items, terms, today):
        items = self._filter(items, terms, lambda j: f"{j.get('title')} {j.get('ad_type')} {j.get('ad_sub_type')}")
        items = sorted(items, key=lambda j: j.get("created_at") or "", reverse=True)[:MAX_ITEMS]
        out = []
        for j in items:
            close = (j.get("closing_date") or "")[:10]
            lines = [f"Advertisement: {j.get('title')}", f"Posted: {(j.get('created_at') or '')[:10]}"]
            if close:
                lines.append(f"Closing date: {close}" + (" (open)" if close >= today else " (closed)"))
            if j.get("apply_link"):
                lines.append(f"Apply at: {j['apply_link']}")
            out.append(Source(f"Job/Position: {j.get('title')}", f"{BASE}/jobs#{j.get('id')}", "live", today,
                              "\n".join(lines)))
        return out

    def _calendar(self, items, terms, today):
        upcoming = sorted((e for e in items if (e.get("end_date") or e.get("start_date") or "")[:10] >= today),
                          key=lambda e: e.get("start_date") or "")
        if not upcoming:
            return []
        lines = []
        for e in upcoming[:20]:
            when = (e.get("start_date") or "")[:10]
            if e.get("end_date"):
                when += f" to {e['end_date'][:10]}"
            kind = (e.get("category") or "").replace("_", " ")
            lines.append(f"- {when}: {e.get('title')} ({kind})" + (f" {e['description']}" if e.get("description") else ""))
        return [Source("IIT Ropar Institute Calendar (upcoming)", f"{BASE}/calendar", "live", today,
                       f"Upcoming institute calendar entries from {today} (restricted holidays are optional):\n"
                       + "\n".join(lines))]
