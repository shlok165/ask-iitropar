"""Evaluate Ask IIT Ropar.

  python eval/run_eval.py                      # routing + retrieval hit@5/MRR + confidence calibration (free)
  python eval/run_eval.py --sweep              # also try HYBRID_ALPHA x TOP_K combinations (free)
  python eval/run_eval.py --answers --provider ollama
  python eval/run_eval.py --answers --provider pollinations --model openai/gpt-5.4-nano --max-cost 0.15
  python eval/run_eval.py --report             # rebuild eval/report.md from saved results

Results go to eval/results/, the summary to eval/report.md, full answers to eval/spotcheck_<model>.md.
Answer runs use their own state database, so nothing is served from the app's answer cache.
"""
import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

EVAL = config.ROOT / "eval"
RESULTS = EVAL / "results"
QUESTIONS = EVAL / "questions.jsonl"
ALPHAS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
KS = [4, 6, 8, 10]
ABSTAIN = re.compile(r"couldn't find|could not find|can't|cannot|not able|don't have|focused on IIT Ropar", re.I)


def load_questions() -> list[dict]:
    qs = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    for q in qs:
        q.setdefault("answerable", True)
    return qs


def norm_url(u: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", u.lower()).rstrip("/")


def matches(src: dict, expected: list[str]) -> bool:
    """A source answers the question if its URL/title matches any expected item.
    'http…' = exact URL, 'type:live' = source type, anything else = substring of URL or title."""
    for e in expected:
        if e.startswith("type:"):
            if src.get("source_type") == e[5:]:
                return True
        elif e.startswith("http"):
            if norm_url(src["url"]) == norm_url(e):
                return True
        elif e.lower() in src["url"].lower() or e.lower() in src["title"].lower():
            return True
    return False


def first_hit(srcs: list[dict], expected: list[str]) -> int | None:
    return next((i for i, s in enumerate(srcs, 1) if matches(s, expected)), None)


def plain(text: str) -> str:
    """Lower-case with typographic apostrophes/quotes made plain (models write "couldn’t")."""
    return text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def must_include_ok(text: str, groups: list[str]) -> bool:
    return all(any(alt.lower() in plain(text) for alt in g.split("|")) for g in groups)


def save(name: str, obj) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --- retrieval & routing ------------------------------------------------------------------
def eval_retrieval(sweep: bool) -> None:
    from app.context import ContextBuilder
    from app.router import route

    builder = ContextBuilder()
    rows = []
    for q in load_questions():
        t0 = time.time()
        r = route(q["question"], builder.tables)
        ctx = builder.gather(q["question"])
        pipe = [{"title": s.title, "url": s.url, "source_type": s.source_type} for s in ctx.sources]
        hits = builder.searcher.search(q["question"], k=10)
        hyb = [{"title": h.title, "url": h.url, "source_type": h.source_type} for h in hits]
        row = {
            "id": q["id"], "category": q["category"], "question": q["question"], "answerable": q["answerable"],
            "route_expected": q["route"], "route": r.kind, "route_reason": r.reason,
            "final_route": ctx.route.kind,
            "pipeline_rank": first_hit(pipe, q["expected"]) if q["expected"] else None,
            "hybrid_rank": first_hit(hyb, q["expected"]) if q["expected"] else None,
            "confidence": builder.searcher.confidence(hits[:config.TOP_K]),
            "pipeline_sources": [s["url"] for s in pipe[:6]],
            "seconds": round(time.time() - t0, 2),
        }
        rows.append(row)
        print(f"{row['id']:6} route {r.kind:9} {'ok ' if r.kind == q['route'] else 'BAD'} "
              f"pipe@{row['pipeline_rank']} hyb@{row['hybrid_rank']} conf={row['confidence']:.2f}")
    save("retrieval.json", rows)

    if sweep:
        grid = []
        targets = [q for q in load_questions() if q["expected"] and q["route"] == "retrieval"]
        for alpha in ALPHAS:
            ranks = []
            for q in targets:
                hits = builder.searcher.search(q["question"], k=max(KS), alpha=alpha)
                ranks.append(first_hit([{"title": h.title, "url": h.url, "source_type": h.source_type}
                                        for h in hits], q["expected"]))
            grid.append({"alpha": alpha, **{f"hit@{k}": sum(1 for r in ranks if r and r <= k) / len(ranks) for k in KS},
                         "mrr@10": sum(1 / r for r in ranks if r) / len(ranks), "n": len(ranks)})
            print(f"alpha={alpha:.2f} " + " ".join(f"hit@{k}={grid[-1][f'hit@{k}']:.2f}" for k in KS)
                  + f" mrr={grid[-1]['mrr@10']:.3f}")
        save("sweep.json", grid)


# --- answers --------------------------------------------------------------------------------
def eval_answers(provider_name: str, model: str | None, max_cost: float, limit: int | None, suffix: str = "") -> None:
    from app.answer import AnswerEngine
    from app.llm import get_provider
    from app.state import RateLimiter, Store

    provider = get_provider(provider_name)
    if model:
        provider.model = model
    tag = f"{provider.name}_{provider.model}{'_' + suffix if suffix else ''}".replace("/", "-").replace(":", "-")
    db = RESULTS / f"state_{tag}.db"
    db.unlink(missing_ok=True)  # fresh cache, so every question really runs
    engine = AnswerEngine(provider=provider, fallback=None, store=Store(db), limiter=RateLimiter(10**6))
    old_cap, config.DAILY_SPEND_CAP = config.DAILY_SPEND_CAP, max_cost + 1  # the eval has its own cost guard

    rows, spent = [], 0.0
    questions = load_questions()[:limit] if limit else load_questions()
    for q in questions:
        if spent >= max_cost:
            print(f"Stopping: spent {spent:.4f} >= --max-cost {max_cost}")
            break
        t0, first, text, meta, done, error = time.time(), None, "", {}, {}, None
        for ev in engine.answer(q["question"]):
            if ev["type"] == "meta":
                meta = ev
            elif ev["type"] == "delta":
                first = first or time.time()
                text += ev["text"]
            elif ev["type"] == "done":
                done = ev
            elif ev["type"] == "error":
                error = ev["message"]
        spent += done.get("cost") or 0.0
        srcs = meta.get("sources", [])
        cited = {int(n) for n in re.findall(r"\[(\d{1,2})\]", text)}
        cited_srcs = [s for s in srcs if s["n"] in cited]
        row = {
            "id": q["id"], "category": q["category"], "question": q["question"], "answerable": q["answerable"],
            "route": meta.get("route"), "answer": text, "error": error,
            "sources": [{"n": s["n"], "title": s["title"], "url": s["url"]} for s in srcs],
            "must_include_ok": must_include_ok(text, q["must_include"]),
            "cites_expected": any(matches(s, q["expected"]) for s in cited_srcs) if q["expected"] else None,
            "has_citation": bool(cited_srcs),
            "abstained": bool(ABSTAIN.search(plain(text))),
            "web_label_misused": meta.get("route") != "web" and "From the web" in text,
            "cost": done.get("cost") or 0.0, "usage": done.get("usage"),
            "seconds": round(time.time() - t0, 1), "first_token_s": round((first or time.time()) - t0, 1),
        }
        rows.append(row)
        print(f"{q['id']:6} {'PASS' if row['must_include_ok'] else 'miss'} cite={row['cites_expected']} "
              f"cost={row['cost']:.5f} total={spent:.4f} {row['seconds']}s")
    config.DAILY_SPEND_CAP = old_cap
    save(f"answers_{tag}.json", {"provider": provider.name, "model": provider.model, "rows": rows,
                                 "total_cost": spent, "ts": time.strftime("%Y-%m-%d %H:%M"),
                                 "top_k": config.TOP_K, "tag": tag})
    write_spotcheck(tag, provider, rows)


def write_spotcheck(tag: str, provider, rows: list[dict]) -> None:
    out = [f"# Spot-check: {provider.name} {provider.model}\n",
           "Every answer from the eval run, with its sources. ✅ = contains the expected facts.\n"]
    for r in rows:
        mark = "✅" if r["must_include_ok"] else "❌"
        out.append(f"\n## {mark} {r['id']}: {r['question']}\n")
        out.append(f"*route: {r['route']} · cost: {r['cost']:.5f} · {r['seconds']} s*\n")
        out.append("\n" + (r["answer"] or f"(error: {r['error']})").strip() + "\n")
        if r["sources"]:
            out.append("\n<details><summary>Sources sent to the model</summary>\n\n"
                       + "\n".join(f"{s['n']}. {s['title']} — {s['url']}" for s in r["sources"]) + "\n\n</details>\n")
    (EVAL / f"spotcheck_{tag}.md").write_text("".join(out), encoding="utf-8")


# --- report ----------------------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def write_report() -> None:
    out = ["# Ask IIT Ropar: evaluation report\n",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `eval/questions.jsonl` "
           f"({len(load_questions())} questions). Settings: TOP_K={config.TOP_K}, "
           f"HYBRID_ALPHA={config.HYBRID_ALPHA}, MIN_RELEVANCE={config.MIN_RELEVANCE}.\n"]
    rows = load("retrieval.json")
    if rows:
        ans = [r for r in rows if r["answerable"] and r["pipeline_rank"] is not None or
               (r["answerable"] and r["route_expected"] != "refuse")]
        with_exp = [r for r in rows if r["answerable"]]
        route_ok = sum(r["route"] == r["route_expected"] for r in rows)
        p_hit = [r for r in with_exp if r["pipeline_rank"] and r["pipeline_rank"] <= 5]
        h_rows = [r for r in with_exp if r["route_expected"] == "retrieval"]
        h_hit = [r for r in h_rows if r["hybrid_rank"] and r["hybrid_rank"] <= 5]
        mrr = sum(1 / r["pipeline_rank"] for r in with_exp if r["pipeline_rank"]) / len(with_exp)
        out += ["\n## Retrieval and routing (no LLM, free)\n",
                "> These numbers are optimistic: the questions were written from the scraped data, and the router was "
                "fixed against this same set (3 facilities questions were misrouted at first). A fresh, held-out set "
                "of real student questions would give a fairer estimate.\n\n",
                "| Metric | Result |\n|---|---|\n",
                f"| Routing accuracy | {route_ok}/{len(rows)} ({pct(route_ok / len(rows))}) |\n",
                f"| Full pipeline hit@5 (answerable questions) | {len(p_hit)}/{len(with_exp)} ({pct(len(p_hit) / len(with_exp))}) |\n",
                f"| Full pipeline MRR | {mrr:.3f} |\n",
                f"| Hybrid search alone hit@5 (retrieval-route questions) | {len(h_hit)}/{len(h_rows)} ({pct(len(h_hit) / len(h_rows))}) |\n"]
        cats = defaultdict(list)
        for r in with_exp:
            cats[r["category"]].append(r)
        out.append("\n| Category | Questions | Pipeline hit@5 | Routed as expected |\n|---|---|---|---|\n")
        for c, rs in sorted(cats.items()):
            out.append(f"| {c} | {len(rs)} | {sum(1 for r in rs if r['pipeline_rank'] and r['pipeline_rank'] <= 5)}/{len(rs)}"
                       f" | {sum(r['route'] == r['route_expected'] for r in rs)}/{len(rs)} |\n")
        bad_route = [r for r in rows if r["route"] != r["route_expected"]]
        misses = [r for r in with_exp if not (r["pipeline_rank"] and r["pipeline_rank"] <= 5)]
        if bad_route:
            out.append("\n**Misrouted:**\n\n" + "".join(
                f"- `{r['id']}` {r['question']}: expected {r['route_expected']}, got {r['route']} ({r['route_reason']})\n"
                for r in bad_route))
        if misses:
            out.append("\n**Retrieval misses (expected source not in top 5):**\n\n" + "".join(
                f"- `{r['id']}` {r['question']} (route {r['final_route']}; got: "
                f"{', '.join(u.split('//')[-1][:60] for u in r['pipeline_sources'][:3])})\n" for r in misses))
        conf_in = [r["confidence"] for r in rows if r["answerable"] and r["route_expected"] == "retrieval"]
        conf_out = [r["confidence"] for r in rows if r["category"] == "out-of-scope"]
        if conf_in and conf_out:
            out += ["\n### Confidence calibration (best cosine similarity)\n",
                    f"- Answerable retrieval questions: min {min(conf_in):.2f}, median {statistics.median(conf_in):.2f}\n",
                    f"- Out-of-scope questions: {', '.join(f'{c:.2f}' for c in conf_out)}\n",
                    f"- MIN_RELEVANCE is {config.MIN_RELEVANCE}: offers a web search for "
                    f"{sum(c < config.MIN_RELEVANCE for c in conf_out)}/{len(conf_out)} out-of-scope and "
                    f"{sum(c < config.MIN_RELEVANCE for c in conf_in)}/{len(conf_in)} answerable questions.\n"]
    grid = load("sweep.json")
    if grid:
        out += ["\n## Tuning sweep (retrieval-route questions, hybrid search alone)\n",
                "`HYBRID_ALPHA`: 0 = keyword (BM25) only, 1 = embeddings only. hit@k = expected source within the "
                "first k chunks (k = `TOP_K`, the chunks sent to the LLM).\n\n",
                "| alpha | " + " | ".join(f"hit@{k}" for k in KS) + " | MRR@10 |\n",
                "|---|" + "---|" * (len(KS) + 1) + "\n"]
        for g in grid:
            out.append(f"| {g['alpha']:.2f} | " + " | ".join(pct(g[f'hit@{k}']) for k in KS) + f" | {g['mrr@10']:.3f} |\n")

    runs = sorted(RESULTS.glob("answers_*.json")) if RESULTS.exists() else []
    if runs:
        out += ["\n## Answer quality (one run per model)\n",
                "| Provider / model | Answers | Facts correct | Cites the expected source | Says \"couldn't find\" / refuses when it should | "
                "Mislabels official answers as web | Total cost | Cost per answer | Avg time |\n",
                "|---|---|---|---|---|---|---|---|---|\n"]
        qs = {q["id"]: q for q in load_questions()}
        for p in runs:
            run = json.loads(p.read_text(encoding="utf-8"))
            rs = run["rows"]
            for r in rs:  # re-score from the saved answer text with the current checks
                r["must_include_ok"] = must_include_ok(r["answer"], qs[r["id"]]["must_include"])
                r["abstained"] = bool(ABSTAIN.search(plain(r["answer"])))
            ok = sum(r["must_include_ok"] for r in rs)
            cite = [r for r in rs if r["cites_expected"] is not None]
            neg = [r for r in rs if not r["answerable"]]
            unit = "pollen" if run["provider"] == "pollinations" else "free"
            cost = f"{run['total_cost']:.4f} {unit}" if unit != "free" else "free (local)"
            per = f"{run['total_cost'] / len(rs):.5f}" if unit != "free" else "0"
            out.append(f"| {run['provider']} `{run['model']}` (TOP_K={run.get('top_k', '?')}) | {len(rs)} | "
                       f"{ok}/{len(rs)} ({pct(ok / len(rs))}) | "
                       f"{sum(r['cites_expected'] for r in cite)}/{len(cite)} | {sum(r['abstained'] for r in neg)}/{len(neg)} | "
                       f"{sum(r['web_label_misused'] for r in rs)} | {cost} | {per} | "
                       f"{statistics.mean(r['seconds'] for r in rs):.1f} s |\n")
        out.append("\nFull answers for spot-checking: " + ", ".join(
            f"[{p.stem.replace('answers_', '')}](spotcheck_{p.stem.replace('answers_', '')}.md)" for p in runs) + "\n")
        for p in runs:
            run = json.loads(p.read_text(encoding="utf-8"))
            failed = [r for r in run["rows"] if not must_include_ok(r["answer"], qs[r["id"]]["must_include"])]
            if failed:
                out.append(f"\n**{run['model']}: answers missing expected facts:** " +
                           ", ".join(f"`{r['id']}`" for r in failed) + "\n")
    verdict = EVAL / "verdict.md"  # hand-written conclusions from spot-checking the answers
    if verdict.exists():
        out.append(verdict.read_text(encoding="utf-8"))
    (EVAL / "report.md").write_text("".join(out), encoding="utf-8")
    print(f"Wrote {EVAL / 'report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true", help="try HYBRID_ALPHA x TOP_K combinations")
    ap.add_argument("--answers", action="store_true", help="generate and check answers with an LLM")
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model")
    ap.add_argument("--max-cost", type=float, default=0.4, help="stop an answer run once it has spent this much")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--tag", default="", help="suffix for the result file, to keep runs with different settings apart")
    ap.add_argument("--report", action="store_true", help="only rebuild eval/report.md")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.answers:
        eval_answers(args.provider, args.model, args.max_cost, args.limit, args.tag)
    elif not args.report:
        eval_retrieval(args.sweep)
    write_report()
