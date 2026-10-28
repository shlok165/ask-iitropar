# Ask IIT Ropar: evaluation report
Generated 2026-10-22 16:43 from `eval/questions.jsonl` (49 questions). Settings: TOP_K=6, HYBRID_ALPHA=0.5, MIN_RELEVANCE=0.65.

## Retrieval and routing (no LLM, free)
> These numbers are optimistic: the questions were written from the scraped data, and the router was fixed against this same set (3 facilities questions were misrouted at first). A fresh, held-out set of real student questions would give a fairer estimate.

| Metric | Result |
|---|---|
| Routing accuracy | 49/49 (100%) |
| Full pipeline hit@5 (answerable questions) | 44/44 (100%) |
| Full pipeline MRR | 0.905 |
| Hybrid search alone hit@5 (retrieval-route questions) | 28/28 (100%) |

| Category | Questions | Pipeline hit@5 | Routed as expected |
|---|---|---|---|
| academics | 5 | 5/5 | 5/5 |
| administration | 3 | 3/3 | 3/3 |
| admissions | 5 | 5/5 | 5/5 |
| departments | 3 | 3/3 | 3/3 |
| facilities | 3 | 3/3 | 3/3 |
| faculty | 4 | 4/4 | 4/4 |
| fests | 3 | 3/3 | 3/3 |
| general | 2 | 2/2 | 2/2 |
| hostels | 4 | 4/4 | 4/4 |
| news-events | 2 | 2/2 | 2/2 |
| placements | 3 | 3/3 | 3/3 |
| research | 3 | 3/3 | 3/3 |
| student-life | 4 | 4/4 | 4/4 |

### Confidence calibration (best cosine similarity)
- Answerable retrieval questions: min 0.68, median 0.76
- Out-of-scope questions: 0.64, 0.57, 0.80
- MIN_RELEVANCE is 0.65: offers a web search for 2/3 out-of-scope and 0/28 answerable questions.

## Tuning sweep (retrieval-route questions, hybrid search alone)
`HYBRID_ALPHA`: 0 = keyword (BM25) only, 1 = embeddings only. hit@k = expected source within the first k chunks (k = `TOP_K`, the chunks sent to the LLM).

| alpha | hit@4 | hit@6 | hit@8 | hit@10 | MRR@10 |
|---|---|---|---|---|---|
| 0.00 | 89% | 89% | 89% | 96% | 0.746 |
| 0.25 | 93% | 96% | 96% | 96% | 0.769 |
| 0.40 | 96% | 96% | 96% | 96% | 0.810 |
| 0.50 | 96% | 96% | 96% | 96% | 0.815 |
| 0.60 | 93% | 93% | 96% | 96% | 0.817 |
| 0.75 | 93% | 93% | 96% | 96% | 0.817 |
| 1.00 | 96% | 96% | 100% | 100% | 0.799 |

## Answer quality (one run per model)
| Provider / model | Answers | Facts correct | Cites the expected source | Says "couldn't find" / refuses when it should | Mislabels official answers as web | Total cost | Cost per answer | Avg time |
|---|---|---|---|---|---|---|---|---|
| ollama `qwen2.5:7b` (TOP_K=4) | 49 | 46/49 (94%) | 40/44 | 4/5 | 1 | free (local) | 0 | 13.0 s |
| ollama `qwen2.5:7b` (TOP_K=6) | 49 | 48/49 (98%) | 40/44 | 4/5 | 2 | free (local) | 0 | 14.9 s |
| pollinations `openai/gpt-5.4-mini` (TOP_K=6) | 49 | 49/49 (100%) | 42/44 | 5/5 | 0 | 0.1278 pollen | 0.00261 | 5.0 s |
| pollinations `openai/gpt-5.4-nano` (TOP_K=6) | 49 | 49/49 (100%) | 39/44 | 5/5 | 0 | 0.0331 pollen | 0.00067 | 4.7 s |

Full answers for spot-checking: [ollama_qwen2.5-7b_k4](spotcheck_ollama_qwen2.5-7b_k4.md), [ollama_qwen2.5-7b_k6](spotcheck_ollama_qwen2.5-7b_k6.md), [pollinations_openai-gpt-5.4-mini](spotcheck_pollinations_openai-gpt-5.4-mini.md), [pollinations_openai-gpt-5.4-nano](spotcheck_pollinations_openai-gpt-5.4-nano.md)

**qwen2.5:7b: answers missing expected facts:** `fes-1`, `pla-3`, `oos-2`

**qwen2.5:7b: answers missing expected facts:** `oos-2`

## Verdict

**Use `openai/gpt-5.4-mini` as the default.** It got every expected fact (49/49), cited the right source most often (42/44), refused all student lookups, said "couldn't find" for all out-of-scope questions, and never mislabelled official answers as web answers. At about 0.0026 pollen per answer, the 6-pollen key covers roughly 2,300 answers (fewer with the answer cache saving repeats).

**`gpt-5.4-nano` is 4× cheaper (0.0007 pollen per answer) but makes real mistakes the keyword checks miss.** On "When is Zeitgeist?" it answered "13–15 March", taking the old fest-site date over the newer official tender (30 Oct – 1 Nov 2026), while mini picked the tender date and flagged the conflict. It also cited the expected source less often (39/44). It's a reasonable fallback if the budget runs low, not the default.

**qwen2.5:7b (Ollama, free) is good enough for development** (48/49 facts at TOP_K=6) but slower (15 s), and it sometimes labels official answers as "From the web" or speculates.

**Tuning decisions** (from the sweep and the Ollama runs):
- `HYBRID_ALPHA=0.5`: hybrid beats keyword-only (hit@4 89% → 96%) and is at the top of the MRR plateau (0.4–0.75).
- `TOP_K=6`: the expected source is usually in the first 4 chunks, but with 4 chunks qwen lost two facts (46/49 vs 48/49) for a 22% input-token saving. Answer completeness wins.
- `MIN_RELEVANCE=0.65`: bge-small similarity scores run high; 0.65 offers a web search for 2 of 3 out-of-scope questions and none of the answerable ones. The third ("hostel fee at IIT Bombay", 0.80) is too close to our own hostel-fee pages for a threshold to catch, and the model correctly says it can't find it.

**Total spent on these two runs: 0.161 pollen** (estimate was ~0.20).
