
## Verdict

**Use `openai/gpt-5.4-mini` as the default.** It got every expected fact (49/49), cited the right source most often (42/44), refused all student lookups, said "couldn't find" for all out-of-scope questions, and never mislabelled official answers as web answers. At about 0.0026 pollen per answer, the 6-pollen key covers roughly 2,300 answers (fewer with the answer cache saving repeats).

**`gpt-5.4-nano` is 4× cheaper (0.0007 pollen per answer) but makes real mistakes the keyword checks miss.** On "When is Zeitgeist?" it answered "13–15 March", taking the old fest-site date over the newer official tender (30 Oct – 1 Nov 2026), while mini picked the tender date and flagged the conflict. It also cited the expected source less often (39/44). It's a reasonable fallback if the budget runs low, not the default.

**qwen2.5:7b (Ollama, free) is good enough for development** (48/49 facts at TOP_K=6) but slower (15 s), and it sometimes labels official answers as "From the web" or speculates.

**Tuning decisions** (from the sweep and the Ollama runs):
- `HYBRID_ALPHA=0.5`: hybrid beats keyword-only (hit@4 89% → 96%) and is at the top of the MRR plateau (0.4–0.75).
- `TOP_K=6`: the expected source is usually in the first 4 chunks, but with 4 chunks qwen lost two facts (46/49 vs 48/49) for a 22% input-token saving. Answer completeness wins.
- `MIN_RELEVANCE=0.65`: bge-small similarity scores run high; 0.65 offers a web search for 2 of 3 out-of-scope questions and none of the answerable ones. The third ("hostel fee at IIT Bombay", 0.80) is too close to our own hostel-fee pages for a threshold to catch, and the model correctly says it can't find it.

**Total spent on these two runs: 0.161 pollen** (estimate was ~0.20).
