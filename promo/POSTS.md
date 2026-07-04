# QuantOne — launch posts

## Reddit (r/LocalLLaMA) — post this one FIRST

**Title:** I ran 30,000 generations to test whether quantization breaks structured output. Q8 is free. Q3 makes models stop refusing impossible tool calls.

**Body:**

Everyone repeats "Q4 is basically free" but I couldn't find rigorous numbers for the thing agents actually need: valid JSON and correct tool calls. So I measured it.

**Setup:** 5 small models (Llama-3.2-3B, Qwen2.5-3B, Gemma-2-2B, Phi-3.5-mini, SmolLM2-1.7B) × 4 quants (FP16/Q8_0/Q4_K_M/Q3_K_M) × 500 machine-checkable tasks × 3 seeds = 30,000 generations, all llama.cpp on free Kaggle/Colab T4s. Total cost: $0. No LLM judges anywhere — every task is scored by a deterministic validator (JSON parse, schema compliance, exact tool + argument match, should-not-call). Differences only count when a paired bootstrap 95% CI excludes zero.

**Findings:**

- **Q8_0 is statistically free.** Zero significant regressions across all 75 comparisons. Stop paying the FP16 VRAM tax for structured output.
- **Q4_K_M is nearly free.** A few small scattered effects, no consistent degradation.
- **Q3_K_M breaks things.** Schema compliance drops significantly in 3 of 5 models (Qwen2.5-3B: 83.5% → 65.0%).
- **The scary one:** at Q3, two models largely lose the ability to say "none of these tools apply." Gemma-2-2B's correct-decline rate: 83% → 40%. Phi-3.5's: 39% → 0%. The quantized model grabs a tool even when no tool fits — the exact failure that silently corrupts agent pipelines.
- Bonus methods finding: greedy (temp=0) decoding is NOT deterministic on real hardware — 12–27% of cells produced different outputs across identical-config runs (thread/cuBLAS reduction order flips near-ties).

Also caught a fun measurement trap: Phi answers correctly then keeps rambling, and its rambling rate *correlates with quant level* — a strict parser manufactured a fake "Q3 improves Phi" result until we fixed the scoring to extract the first JSON value. Write-up has the full story.

Everything is reproducible: dataset with all 30k raw outputs [HF link], interactive results table [Space link], code + write-up [GitHub link]. Every GGUF sha256, generation config, and RNG seed is pinned. A fresh-session repro script regenerates any cell from scratch.

Limitations: ≤3.8B models, T4, llama.cpp, one task suite. Q3 might behave differently at 7B+ — that's future work.

---

## LinkedIn

I spent two weeks and $0 of compute answering a question every LLM deployment team hand-waves: **does quantization break structured output?**

Quantization (shrinking models from FP16 to 8/4/3-bit) is how everyone fits LLMs on affordable hardware. The folklore says it's "basically free." Nobody I could find had measured that claim rigorously for the capability that actually matters in production — emitting valid JSON and correct tool calls.

So I built QuantOne: 5 open models × 4 quantization levels × 500 machine-checkable tasks × 3 seeds = 30,000 generations, run entirely on free-tier GPUs, scored with deterministic validators (no "LLM judges"), and aggregated with paired bootstrap confidence intervals.

What the data says:
✅ 8-bit is statistically free — zero significant regressions in 75 comparisons
⚠️ 4-bit is nearly free
❌ 3-bit measurably breaks schema compliance in 3 of 5 models
🚨 Most striking: at 3-bit, two model families largely stop *refusing* impossible requests — Gemma-2's correct-decline rate fell from 83% to 40%, Phi-3.5's from 39% to 0%. The compressed model calls a tool even when none applies. If you run agents on aggressively quantized models, that failure mode is silent.

The project also surfaced a measurement lesson I'll carry forward: one model emits correct answers then keeps rambling, at a rate that *correlated with quantization level* — a strict parser turned that artifact into a fake "quantization improves accuracy" result. Catching and fixing that confound (with zero re-runs, thanks to separating generation from scoring) was the most instructive part of the study.

Dataset (all 30k raw outputs), interactive leaderboard, code, and write-up in the comments. Everything is pinned and reproducible — a fresh-machine script regenerates any cell of the matrix and checks it against the published numbers.

#LLM #Quantization #MLEngineering #OpenSource #Evaluation

*(Put the three links in the FIRST COMMENT, not the post body — LinkedIn suppresses posts with external links.)*

---

## X/Twitter thread (optional, pairs with the infographic)

1/ "Q4 is basically free" — everyone says it, nobody measures it for the thing agents need: valid JSON + correct tool calls. I ran 30,000 generations to find out. Results 🧵

2/ Setup: 5 small models × FP16/Q8/Q4/Q3 × 500 machine-checkable tasks × 3 seeds. No LLM judges — deterministic validators only. Paired bootstrap 95% CIs. Free-tier GPUs. $0 total.

3/ Q8: statistically FREE. 0/75 significant regressions. Q4: nearly free. Q3: schema compliance drops in 3/5 models.

4/ The scary result: at Q3, models stop REFUSING. Gemma-2 correct-decline: 83%→40%. Phi-3.5: 39%→0%. The quantized model grabs a tool even when none fits. Silent agent corruption.

5/ Bonus: temp=0 is not deterministic on real hardware — 12-27% of cells differed across identical runs. Your "greedy" outputs are noisier than you think.

6/ Everything's open: 30k raw outputs on HF, interactive leaderboard, pinned hashes, fresh-session repro script. [links]

---

## Distribution checklist (beyond the three posts)

1. **HF dataset card + Space cross-links** — done automatically by the upload script; make sure the Space README links the dataset and GitHub.
2. **Hacker News** — "Show HN: I measured whether quantization breaks LLM structured output (30k generations, CIs)". Post the GitHub repo, morning US time. HN loves rigor + $0 budget stories.
3. **llama.cpp GitHub Discussions** — post in Show & Tell; this is literally their user base's weekly question. Link the write-up, invite scrutiny of the methodology.
4. **HF community post** (huggingface.co/posts) — short version of the LinkedIn post with the headline chart.
5. **Reply ammunition**: when people ask "what about 7B/70B?", "what about imatrix quants?", "what about GPTQ/AWQ?" — the answer is "great question, it's in future-work.md; the harness is reusable, PRs welcome." Every objection is a future project, not a threat.
6. **Timing**: Reddit first (it generates the organic discussion), HN same day, LinkedIn next morning, X alongside Reddit. SamplerLab's post lands ~10 days later and cites this one — the "lab series" effect compounds.
