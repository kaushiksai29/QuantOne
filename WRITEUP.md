# Does quantization break structured output? 30,000 generations say: Q8 no, Q3 yes.

**TL;DR.** We measured JSON/tool-call reliability across FP16 → Q8_0 → Q4_K_M
→ Q3_K_M for five small instruction models (1.7B–3.8B), 500 machine-checkable
tasks × 3 seeds each — 30,000 generations on free-tier Kaggle/Colab T4s,
scored deterministically (no LLM judges), aggregated with paired bootstrap
95% CIs. **Q8_0 showed zero statistically significant regressions across all
75 comparisons. Q4_K_M is nearly free. Q3_K_M measurably degrades schema
compliance in 3 of 5 models and — the sharpest result — collapses the
ability to decline: two model families largely stop refusing tool calls that
no available tool can satisfy.**

## Method (full repro detail)

- **Models** (single GGUF source repo per model; sha256 of every file pinned
  in `matrix.yaml` before first run): Llama-3.2-3B-Instruct (unsloth),
  Qwen2.5-3B-Instruct (bartowski), gemma-2-2b-it (second-state),
  Phi-3.5-mini-instruct (second-state), SmolLM2-1.7B-Instruct (bartowski).
- **Tasks**: 500, generated programmatically from a fixed seed
  (`tasks/generate_tasks.py`, byte-reproducible): 350 schema-constrained JSON
  tasks (nesting depth 1–4; enums, ISO dates, ranged numbers, arrays, nested
  objects), 120 tool-call tasks (2–5 distractors, argument values stated
  verbatim), 30 should-not-call tasks.
- **Generation**: temp=0, max_tokens=512, GGUF-embedded chat templates,
  llama-cpp-python 0.3.19, pipeline_version 1. Content-addressed result files
  make every run idempotent and resumable.
- **Scoring**: deterministic validators only — JSON parse, Draft 2020-12
  schema validation, exact tool selection, exact argument match, correct
  decline. Parsing accepts the first complete JSON value and ignores trailing
  commentary (see "Confound" below).
- **Stats**: per (model, quant, metric): seeds averaged per prompt, then
  paired bootstrap over the 500 prompt_ids (10,000 resamples) for every
  quant-vs-FP16 delta. Significant iff the 95% CI excludes zero.

## Findings

1. **Q8_0 is statistically free.** Not one of the 25 model×metric comparisons
   showed a significant Q8-vs-FP16 delta. If you serve these models with
   llama.cpp, Q8 costs nothing measurable for structured output.
2. **Q4_K_M is nearly free.** Scattered small effects in both directions
   (largest regression: qwen tool_selection −0.031; gemma schema −0.044;
   phi correct_decline −0.122), no consistent cross-model degradation.
3. **Q3_K_M is where reliability breaks.** Schema compliance drops
   significantly in qwen25_3b (−0.186, CI [−0.24, −0.13]), gemma2_2b
   (−0.114) and smollm2_17b (−0.050).
4. **The decline-collapse (headline).** At Q3, gemma2_2b's correct-decline
   rate falls 0.83 → 0.40 (−0.433, CI [−0.60, −0.27]) and phi35_mini's
   0.39 → 0.00 (−0.389, CI [−0.56, −0.22]). Heavily quantized models grab a
   tool even when none applies — precisely the failure that silently corrupts
   agent pipelines. Two independent model families showing the same failure
   direction suggests this is a quantization effect, not a model quirk.
5. **Model quality dominates quant level.** Llama-3.2-3B is weak at deeply
   nested JSON at every precision (schema compliance ~0.27–0.35, often
   emitting truncated JSON and stopping); SmolLM2 never declines at any
   precision (0.00 across the board — a model trait, not a quant effect).
   Choosing the right model matters more than choosing FP16 over Q8/Q4.

## A confound we caught (and how)

Phi-3.5 emits a correct JSON answer, then keeps talking — inventing further
user turns until the token cap. Its trailing-text rate *correlates with
quant level* (FP16 53% → Q3 8%), so a strict "output must be exactly one
JSON document" parser manufactured a fake "Q3 improves phi" result. The fix:
accept the first complete JSON value (via `json.raw_decode`) and ignore
trailing text. Models that don't ramble are byte-unchanged by this rule.
Because scoring is a separate stage from generation, the fix required zero
re-inference — re-scoring 30k outputs took seconds.

## A methods note on "seeds"

Greedy (temp=0) decoding is not bit-stable on real inference stacks:
12–27% of (quant, prompt) cells produced different outputs across seeds
(thread/cuBLAS reduction order flips argmax on near-ties). Our three "seeds"
therefore function as replicates over inference noise, which the
seed-averaging step absorbs. Claims of exact determinism at temp=0 on GPUs
should generally be distrusted.

## Limitations

T4-only; ≤3.8B models; one capability suite (structured output / tool
calls); one GGUF source repo per model (quantizer provenance differs across
models but never within one); llama.cpp inference specifically; lenient
parse rule scores "usable output," while strict no-trailing-text
instruction-following is a different (harsher) metric.

## Reproduce

```
pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pytest                                   # 42 tests: validators, generator, bootstrap
python tasks/generate_tasks.py           # byte-identical tasks.jsonl
python scripts/repro_cell.py --model qwen25_3b --quant Q3_K_M   # one cell from scratch
```

All 30,000 raw outputs + scores + summary are in the HF dataset
(`scripts/upload_dataset.py`). Every GGUF sha256, the task-generation seed,
the bootstrap seed, and pipeline_version are in `matrix.yaml` / the code.
