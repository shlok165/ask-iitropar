# Ask IIT Ropar

A chatbot that answers IIT Ropar students' questions from the institute's official information, with sources: departments, professors, clubs, fests, hostels, academics, admissions, placements and facilities.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -r requirements-app.txt
python -m app                                         # builds the index on first run (~10 min), then http://localhost:7860
```

Settings come from `.env` (real environment variables override it). The main ones:

| Setting | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` | `pollinations` | `pollinations`, `ollama` (free, local) or `fake` (tests) |
| `POLLINATIONS_API_KEY`, `POLLINATIONS_MODEL` | `openai/gpt-5.4-mini` | answering model; see `eval/report.md` for mini vs nano |
| `OLLAMA_MODEL` | `qwen2.5:7b` | local model (`ollama pull qwen2.5:7b`) |
| `DAILY_SPEND_CAP` | `1.0` | pollen per day; over it, answers come from `BUDGET_FALLBACK_PROVIDER` (Ollama) or a "limit reached" message |
| `TOP_K`, `HYBRID_ALPHA`, `MIN_RELEVANCE` | `6`, `0.5`, `0.65` | retrieval settings, tuned in `eval/report.md` |
| `RATE_LIMIT_PER_MINUTE` | `20` | per IP |

All settings and their defaults are in `app/config.py`. To develop without spending pollen, run with `LLM_PROVIDER=ollama`.

Other commands:

```bash
python -m app.index                                   # rebuild the search index after re-scraping
python -m app.answer "When is Zeitgeist?" --provider ollama   # ask from the command line
python -m pytest tests                                # unit tests (no network, no LLM)
python eval/run_eval.py --sweep                       # routing + retrieval eval and tuning sweep (free)
python eval/run_eval.py --answers --provider ollama   # answer-quality eval
```

## How it works

1. **Router** (`app/router.py`, no LLM): refuses lookups of individual students, sends time-sensitive questions to the live IIT Ropar API (`app/live.py`), lookups to the CSV tables (`app/tables.py`), and everything else to hybrid search (`app/search.py`: bge-small embeddings in ChromaDB + BM25, merged by rank fusion). If a path finds nothing, it falls back to hybrid search.
2. **Answer** (`app/answer.py`): `prompts/system_prompt.md` plus numbered sources go to the LLM (`app/llm/`). The answer streams back through `redact()` and is cached; spend is tracked per day.
3. **Web fallback** (`app/websearch.py`): when you ask to search the web, click 👎 or accept the low-confidence offer, it searches iitrpr.ac.in, zeitgeist.org.in and advitiya.org.in first, then the wider web without social media (free `ddgs` search). Answers are labelled "🌐 From the web" and every question is logged to `logs/web_queries.jsonl`, to find gaps in the data.
4. **Logs** in `logs/`: `feedback.jsonl` (👍/👎), `web_queries.jsonl`, `state.db` (answer cache and spend).

## Docker

```bash
docker build -t ask-iitropar .                        # ~15 min: installs CPU PyTorch and builds the index into the image
docker run --rm -p 7860:7860 \
  -v "$PWD/.env:/app/.env:ro" -v "$PWD/logs:/app/logs" \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  ask-iitropar                                        # http://localhost:7860
```

`.env` is mounted read-only (never copied into the image), `logs/` keeps the cache and logs between runs, and `OLLAMA_BASE_URL` lets the container use Ollama on your machine as the over-budget fallback. The same image also runs on a Hugging Face Docker Space (port 7860, non-root user): put the keys in the Space's secrets instead of `.env`.

## Privacy

The bot never looks up or profiles individual students. It names students only where official pages list them as office-bearers, and every answer (including web answers) passes through `redact()`, which removes student entry numbers and mobile numbers. See "Privacy choices" below for what the scraper collects.

---

# Data collection

Scrapers that collect IIT Ropar's **public** web content into RAG-ready files.

## Run

```bash
pip install -r requirements.txt
python -m playwright install chromium
python run_all.py                 # full run (~30-60 min)
python run_all.py --skip api,render   # re-run only the sub-site crawl + build
```

## Pipeline

| Step | Script | What it collects |
|---|---|---|
| 1 | `scraper/fetch_api.py` | The JSON API behind www.iitrpr.ac.in: all faculty (207), staff, 42 departments/sections, committees & members, news/events, institute calendar, admissions, jobs, tenders |
| 2 | `scraper/render_site.py` | Every public page of the main site (it is a React app, so pages are rendered in headless Chromium; accordions and tabs are expanded) — boards, clubs, student council, hostels, academics, admissions, rules, etc. |
| 3 | `scraper/crawl_subsites.py` | Department websites, faculty profile pages, Zeitgeist / Advitiya / Aarohan, BoSA, placements (CDPC/CAPS), library, medical centre, research centres, club sites linked from the boards pages, Wikipedia, and PDFs (handbooks, ordinances, curriculum, annual reports, admission brochures…) |
| 4 | `scraper/build_corpus.py` | Strips menu/footer boilerplate, redacts, de-duplicates, chunks |

## First run (22 Oct 2026)

1,922 unique documents → 7,909 chunks: 655 from the API (207 faculty, 214 staff, 42 departments, committees, news, calendar, admissions, jobs), 1,006 web pages (main site + 200+ sub-sites and faculty homepages), 259 PDFs (handbook, curriculum, hostel rules, ordinances/statutes, annual reports, newsletters, admission brochures, Aarohan rulebook, forms). `data/raw/pdfs` is ~730 MB; delete it if you don't need the originals.

## Output (`data/`)

- `corpus.jsonl` — one cleaned document per line: `id, url, title, category, source, text, fetched_at`
- `chunks.jsonl` — ~350-word chunks (60-word overlap), title-prefixed, ready to embed. Each keeps `url` for citations.
- `faculty.csv`, `staff.csv`, `departments.csv`, `committee_members.csv`, `calendar.csv` — structured tables (good for exact lookups / a SQL or filter tool alongside vector search)
- `markdown/<category>.md` — the corpus as readable markdown, for spot-checking
- `raw/` — raw API JSON, downloaded PDFs, discovered links
- `docs/*.jsonl` — per-step outputs before merging

Categories: `faculty, departments, academics, admissions, student-life, fests, hostels, placements, research, administration, facilities, news-events, general`.

## Privacy choices

The assistant is for students, so it should not become a lookup tool for individual students.

- **Not collected:** `/on-roll-students` (full student list), admission selection/shortlist PDFs, department "students"/batch roster pages, login-only portals (AIMS, Samarth, intranet, eOffice).
- **Safety net:** `build_corpus.py` also drops any page that lists 10+ student entry numbers or whose title is a selection/shortlist/merit list, which catches roster pages the URL filters miss (it dropped 22 on the first run). Official board/council office-bearer pages are exempt.
- **Redacted:** student entry numbers (`2023XXX1234`) and mobile numbers are replaced with `[... redacted]` in every document.
- **Kept:** names and roles of students the institute publishes as office-bearers (general secretaries, club representatives, student council), with their official role emails; faculty/staff professional info (designation, office, office phone, institute email, research areas).

## Using it for RAG

1. Embed `chunks.jsonl[].text` (e.g. `bge-small-en`, `text-embedding-3-small`) into a vector store (Chroma, FAISS, Qdrant).
2. Store `url`, `title`, `category` as metadata; filter by category when a question is clearly about one area.
3. For "who teaches X / list professors in Y" questions, query `faculty.csv` directly — tables beat vector search for lists.
4. Re-run the pipeline every few weeks; news, jobs, tenders and admissions change often.
