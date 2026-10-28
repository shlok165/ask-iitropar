"""Step 3: crawl IIT Ropar sub-sites (departments, fests, placements, library, centres, clubs) and PDFs."""
import asyncio
import csv
import io
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup
from pypdf import PdfReader

from common import (DATA, RAW, clean_text, get, is_excluded, make_doc, normalize_url,
                    robots_allowed, save_docs)
from render import render_crawl
from render_site import is_spa

# (url, max_pages, filter_student_rosters)
SEEDS = [
    # Fests & student life
    ("https://www.zeitgeist.org.in/", 80, False),
    ("https://www.advitiya.org.in/", 80, False),
    ("https://www.iitrpr.ac.in/downloads/bosa/#/home", 60, False),
    ("https://www.iitrpr.ac.in/ismp/", 40, False),
    ("https://www.iitrpr.ac.in/enactus/", 30, False),
    ("https://hostel.iitrpr.ac.in/", 40, False),
    ("https://www.iitrpr.ac.in/womensforum/", 20, False),
    # Placements & alumni
    ("https://placements.iitrpr.ac.in/", 60, True),
    ("https://caps.iitrpr.ac.in/", 40, True),
    ("https://alumniconnect.iitrpr.ac.in/", 15, True),
    # Facilities & services
    ("https://www.iitrpr.ac.in/library/", 80, False),
    ("https://iitrpr.ac.in/medical-center/", 30, False),
    ("https://iitrpr.ac.in/crf/", 40, False),
    ("https://iitrpr.ac.in/centralWorkshop/", 20, False),
    ("https://www.iitrpr.ac.in/downloads/HPC/index.html", 20, False),
    ("https://www.iitrpr.ac.in/directory/", 30, False),
    ("https://iitrpr.ac.in/downloads/emergency-contacts.html", 1, False),
    ("https://iitrpr.ac.in/downloads/", 60, False),
    ("https://sites.google.com/iitrpr.ac.in/creche/home", 10, False),
    ("https://sites.google.com/iitrpr.ac.in/ipr", 10, False),
    # Research
    ("https://www.iitrpr.ac.in/datascience/", 40, True),
    ("https://iitrpr.ac.in/creed", 30, True),
    ("https://iitrpr.ac.in/indo-taiwan/", 30, True),
    ("https://iitrpr.ac.in/cdas/", 30, True),
    ("https://www.iitrpr.ac.in/coe-sards/About.html", 30, True),
    ("https://iitrpr.ac.in/oir/", 40, False),
    ("https://www.iitrpr.ac.in/RnD/", 40, True),
    ("https://www.iitrpr.ac.in/tbif/", 30, False),
    ("https://www.iitrpr.ac.in/drif/", 30, False),
    # General background
    ("https://en.wikipedia.org/wiki/Indian_Institute_of_Technology_Ropar", 1, False),
]
SKIP_HOSTS = re.compile(r"samarth|eoffice|intranet|/aims|mail\.|campusmall|facebook|instagram|twitter|x\.com|"
                        r"linkedin|youtube|google\.co|maps\.|pmindia|forms\.gle|docs\.google|whatsapp", re.I)
STUDENT_LIFE_PAGE = re.compile(r"board|student|aarohan|kshitij|bost|club|hostel", re.I)
# Roster pages that list individual students (applied on department / research sites).
ROSTER_RE = re.compile(r"[/=](students?|student[-_]?list|scholars?|phd[-_]?(students|scholars)|ug[-_]?students|"
                       r"pg[-_]?students|b[-_]?tech|m[-_]?tech|m[-_]?sc|b[-_]?sc|batch\d*|alumni|placed[-_]?students)"
                       r"(/|\.|$|[-_&])", re.I)
SKIP_EXT = re.compile(r"\.(jpe?g|png|gif|webp|svg|ico|mp4|mp3|zip|rar|7z|exe|apk|xlsx?|docx?|pptx?|csv|css|js)$", re.I)

MAX_PDFS, MAX_PDF_MB, MAX_PDF_PAGES = 400, 25, 300
PDF_DIR = RAW / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()
pdf_seen: set[str] = set()


def host(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def scope_of(url):
    p = urlparse(url)
    path = p.path
    if "." in path.rsplit("/", 1)[-1]:  # file-like last segment -> its directory
        path = path.rsplit("/", 1)[0] + "/"
    return host(url), path if path not in ("", "/") else "/"


def in_scope(url, scope):
    h, prefix = scope
    return host(url) == h and urlparse(url).path.startswith(prefix.rstrip("/"))


def html_extract(html, url):
    soup = BeautifulSoup(html, "html.parser")
    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "svg", "form", "iframe"]):
        tag.decompose()
    main = soup.find("main") or soup.find(id=re.compile("content|main", re.I)) or soup.body or soup
    return title, clean_text(main.get_text("\n")), links


def fetch_pdf(url, referrer=""):
    """Download a PDF (size-capped) and return a doc with its text, or None."""
    with _lock:
        if url in pdf_seen or len(pdf_seen) >= MAX_PDFS or is_excluded(url):
            return None
        pdf_seen.add(url)
    r = get(url, timeout=60, stream=True)
    if r is None or r.status_code != 200:
        return None
    data, limit = b"", MAX_PDF_MB * 1024 * 1024
    for chunk in r.iter_content(65536):
        data += chunk
        if len(data) > limit:
            print(f"  ! skip large pdf {url}")
            return None
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((pg.extract_text() or "") for pg in reader.pages[:MAX_PDF_PAGES])
    except Exception as e:
        print(f"  ! pdf parse {url}: {e.__class__.__name__}")
        return None
    name = re.sub(r"[^\w.-]+", "_", unquote(urlparse(url).path.rsplit("/", 1)[-1]))[:120] or "file.pdf"
    (PDF_DIR / name).write_bytes(data)
    if len(text.strip()) < 100:  # scanned PDF without a text layer
        return None
    title = name.rsplit(".", 1)[0].replace("_", " ")
    print(f"  [pdf] {url} ({len(reader.pages)} pages)")
    return make_doc(url, f"PDF: {title}", text, "pdf", referrer=referrer, pages=len(reader.pages))


def crawl_static(seed, max_pages, roster_filter):
    """Requests-based BFS within the seed's scope. Returns (docs, pdf_links, needs_js)."""
    scope = scope_of(seed)
    queue, seen, docs, pdfs = [(normalize_url(seed), 0)], {normalize_url(seed)}, [], []
    needs_js = False
    while queue and len(docs) < max_pages:
        url, depth = queue.pop(0)
        if not robots_allowed(url):
            continue
        if "#/" in url:  # hash-router app
            return docs, pdfs, True
        r = get(url, timeout=25)
        if r is None or r.status_code != 200:
            continue
        ctype = r.headers.get("content-type", "")
        if "pdf" in ctype:
            pdfs.append((url, seed))
            continue
        if "html" not in ctype:
            continue
        title, text, links = html_extract(r.text, r.url)
        if len(text) < 250 and len(docs) == 0 and re.search(r'id="(root|app|__next)"|<script[^>]+src', r.text):
            return docs, pdfs, True
        if len(text) >= 80:
            docs.append(make_doc(url, title or url, text, f"site:{host(seed)}"))
            print(f"  [{host(seed)} {len(docs)}] {url}")
        if depth >= 4:
            continue
        for link in links:
            n = normalize_url(link)
            if not n.startswith("http") or n in seen or is_excluded(n) or SKIP_HOSTS.search(n):
                continue
            seen.add(n)
            if n.lower().endswith(".pdf"):
                if host(n).endswith("iitrpr.ac.in") or in_scope(n, scope):
                    pdfs.append((n, url))
            elif in_scope(n, scope) and not SKIP_EXT.search(n) and not is_spa(n) \
                    and not (roster_filter and ROSTER_RE.search(n.split("://", 1)[-1])):
                queue.append((n, depth + 1))
    return docs, pdfs, needs_js


def build_seeds():
    seeds = {normalize_url(u): (u, n, f) for u, n, f in SEEDS}
    # Department websites.
    dept_csv = DATA / "departments.csv"
    if dept_csv.exists():
        for row in csv.DictReader(dept_csv.open(encoding="utf-8-sig")):
            link = (row.get("dept_external_link") or "").strip()
            if link.startswith("http") and not is_spa(link):
                seeds.setdefault(normalize_url(link), (link, 120, True))
    # Faculty personal / profile pages (one page each).
    fac_csv = DATA / "faculty.csv"
    if fac_csv.exists():
        for row in csv.DictReader(fac_csv.open(encoding="utf-8-sig")):
            link = (row.get("profile_link") or "").strip()
            if link.startswith("http") and not SKIP_HOSTS.search(link):
                seeds.setdefault(normalize_url(link), (link, 1, True))
    # Links found while rendering the main site.
    disc_path = RAW / "discovered_links.json"
    pdf_links = []
    if disc_path.exists():
        for link, sources in json.loads(disc_path.read_text(encoding="utf-8")).items():
            if is_excluded(link) or SKIP_HOSTS.search(link) or is_spa(link.replace("://iitrpr", "://www.iitrpr")):
                continue
            if link.lower().endswith(".pdf") or "/api/admission-file/" in link:
                pdf_links.append((link, sources[0]))
            elif any(in_scope(link, scope_of(s[0])) for s in seeds.values()):
                continue
            elif host(link).endswith("iitrpr.ac.in"):
                seeds.setdefault(link, (link, 40, True))
            elif any(STUDENT_LIFE_PAGE.search(s) for s in sources):  # club / fest websites
                seeds.setdefault(link, (link, 25, False))
    return list(seeds.values()), pdf_links


def main():
    seeds, pdf_links = build_seeds()
    print(f"{len(seeds)} seeds, {len(pdf_links)} PDF links from main site")
    docs, js_seeds, all_pdfs = [], [], list(pdf_links)

    with ThreadPoolExecutor(6) as ex:
        for (seed, n, roster), (d, p, js) in zip(seeds, ex.map(lambda s: crawl_static(*s), seeds)):
            docs += d
            all_pdfs += p
            if js:
                js_seeds.append((seed, n, roster))

    # JavaScript-only sites: render them in a headless browser.
    for seed, n, roster in js_seeds:
        print(f"[render] {seed}")
        scope = scope_of(seed)
        allow = (lambda sc, rf: lambda u: in_scope(u, sc) and not SKIP_EXT.search(u) and not u.lower().endswith(".pdf")
                 and not (rf and ROSTER_RE.search(u.split("://", 1)[-1])))(scope, roster)
        pages = asyncio.run(render_crawl([seed], allow, max_pages=n, concurrency=3))
        for pg in pages:
            if len(pg["text"]) >= 80:
                docs.append(make_doc(pg["url"], pg.get("doc_title") or pg["title"], pg["text"], f"site:{host(seed)}"))
            all_pdfs += [(l, pg["url"]) for l in pg["links"] if l.lower().endswith(".pdf") and host(l).endswith("iitrpr.ac.in")]

    save_docs("subsites", docs)

    with ThreadPoolExecutor(4) as ex:
        pdf_docs = [d for d in ex.map(lambda x: fetch_pdf(*x), dict(all_pdfs).items()) if d]
    save_docs("pdfs", pdf_docs)


if __name__ == "__main__":
    main()
