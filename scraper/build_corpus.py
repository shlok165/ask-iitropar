"""Step 4: merge all docs, redact student personal data, de-duplicate, chunk for embedding."""
import hashlib
import json
import re
from collections import Counter
from urllib.parse import unquote

from common import DATA, DOCS, ENTRY_NO_RE, redact

OFFICE_BEARER_URL_RE = re.compile(r"iitrpr\.ac\.in/(boards(/|$)|student-council|student-affairs)")
ROSTER_TITLE_RE = re.compile(r"s(e)?lected|shortlist|provisional(ly)? select|merit list|on[- ]roll", re.I)

CHUNK_WORDS, OVERLAP_WORDS = 350, 60
# Lines that repeat across many pages (menus, footers) are boilerplate.
BOILERPLATE_MIN_PAGES = 25


def load_docs():
    docs = []
    for path in sorted(DOCS.glob("*.jsonl")):
        docs += [json.loads(line) for line in path.open(encoding="utf-8")]
    return docs


def strip_boilerplate(docs):
    line_pages = Counter()
    for d in docs:
        if d["source"].startswith("site:"):
            line_pages.update({ln.strip() for ln in d["text"].splitlines() if len(ln.strip()) > 2})
    common = {ln for ln, n in line_pages.items() if n >= BOILERPLATE_MIN_PAGES}
    for d in docs:
        if d["source"].startswith("site:"):
            d["text"] = "\n".join(ln for ln in d["text"].splitlines() if ln.strip() not in common).strip()
    return docs


def chunk(text):
    """Pack whole lines into ~CHUNK_WORDS chunks, carrying ~OVERLAP_WORDS of trailing lines over."""
    chunks, cur, n = [], [], 0
    for line in (ln for ln in text.splitlines() if ln.strip()):
        w = len(line.split())
        if cur and n + w > CHUNK_WORDS:
            chunks.append("\n".join(cur))
            keep, k = [], 0
            for ln in reversed(cur):
                k += len(ln.split())
                if k > OVERLAP_WORDS:
                    break
                keep.insert(0, ln)
            cur, n = keep, sum(len(ln.split()) for ln in keep)
        if w > CHUNK_WORDS:  # one giant line (e.g. PDF text without newlines)
            ws = line.split()
            chunks += [" ".join(ws[i:i + CHUNK_WORDS]) for i in range(0, len(ws), CHUNK_WORDS - OVERLAP_WORDS)]
            continue
        cur.append(line)
        n += w
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def main():
    docs = strip_boilerplate(load_docs())
    seen, corpus, dropped = set(), [], []
    for d in docs:
        if d["source"] == "pdf":
            d["title"] = "PDF: " + unquote(d["url"].rsplit("/", 1)[-1]).rsplit(".", 1)[0].replace("_", " ")
        # Safety net for student rosters that slipped past the URL filters: drop documents
        # that list many entry numbers or are admission selection lists.
        n_entry = len(ENTRY_NO_RE.findall(d["text"]))
        words = max(len(d["text"].split()), 1)
        office_bearers = OFFICE_BEARER_URL_RE.search(d["url"])  # official club/board office-bearer lists are kept
        if (n_entry >= 10 and not office_bearers and (d["source"] != "pdf" or n_entry / words > 1 / 300))                 or ROSTER_TITLE_RE.search(d["title"]):
            dropped.append(d["url"])
            continue
        d["text"] = redact(d["text"])
        d["title"] = redact(d["title"])
        key = hashlib.sha1(re.sub(r"\W+", "", d["text"].lower()).encode()).hexdigest()
        if len(d["text"]) < 80 or key in seen:
            continue
        seen.add(key)
        corpus.append(d)

    with (DATA / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    n_chunks = 0
    with (DATA / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for d in corpus:
            for i, c in enumerate(chunk(d["text"])):
                f.write(json.dumps({
                    "chunk_id": f"{d['id']}-{i}",
                    "doc_id": d["id"],
                    "title": d["title"],
                    "url": d["url"],
                    "category": d["category"],
                    "source": d["source"],
                    # Title prefix gives each chunk context for retrieval.
                    "text": f"{d['title']}\n{c}",
                }, ensure_ascii=False) + "\n")
                n_chunks += 1

    # Human-browsable markdown, one file per category.
    md_dir = DATA / "markdown"
    md_dir.mkdir(exist_ok=True)
    by_cat = {}
    for d in corpus:
        by_cat.setdefault(d["category"], []).append(d)
    for cat, ds in by_cat.items():
        with (md_dir / f"{cat}.md").open("w", encoding="utf-8") as f:
            for d in ds:
                f.write(f"# {d['title']}\n\nSource: {d['url']}\n\n{d['text']}\n\n---\n\n")

    print(f"{len(docs)} raw docs -> {len(corpus)} unique docs -> {n_chunks} chunks")
    print(f"Dropped {len(dropped)} student-roster/selection-list docs:", *dropped[:30], sep="\n  ")
    print("By category:", dict(Counter(d["category"] for d in corpus).most_common()))
    print("By source:", dict(Counter(d["source"].split(":")[0] for d in corpus).most_common()))


if __name__ == "__main__":
    main()
