# Project: QuantOne — quantization vs structured-output reliability

## What this is
A statistically rigorous benchmark: 5 models × 4 quant levels (FP16/Q8/Q4_K_M/Q3_K_M)
× 500 structured-output tasks × 3 seeds = 30,000 generations, scored with
machine-checkable validators, aggregated with paired bootstrap CIs (10k resamples).
Deliverable: HF dataset (prompts + raw outputs + scores) + static leaderboard Space
+ write-up. Runs on Kaggle free T4. Total budget: $0.

## Non-negotiable rules (never violate, never "improve")
1. NO LLM JUDGES. Every task ships a deterministic, machine-checkable gold validator.
   If a task can't be validated programmatically, delete the task.
2. CONTENT-ADDRESSED RESULTS. Every generation writes to
   results/{sha1(model + quant + prompt_id + seed)}.json. Re-running is always
   idempotent. Scoring reads files; scoring NEVER triggers inference.
3. FROZEN CONFIG. Generation config (temp=0, max_tokens, chat template per family)
   is set once in matrix.yaml and versioned. Changing it invalidates all results —
   if a change is truly needed, bump pipeline_version and rerun everything.
4. FROZEN SCOPE. 5 models, 4 quants, 500 tasks, 3 seeds. Any idea to add models,
   tasks, metrics, or features goes into future-work.md, not into code.
5. SHA256 EVERY GGUF. Record in matrix.yaml before first run. If a hash mismatches
   at run time, abort — do not silently redownload.
6. SEPARATION OF STAGES. generate_tasks → run → score → aggregate → report are
   independent scripts with file-based interfaces. No stage imports another's
   internals. Each is independently re-runnable.
7. KAGGLE CONSTRAINTS. Sessions cap ~12h. Runner must checkpoint-free resume
   purely from the content-addressed results dir (rule 2 gives this for free).
   Never assume a run completes in one session.
8. STATS DISCIPLINE. Differences are reported significant only when the paired
   bootstrap 95% CI excludes zero. Report CIs on everything. A null result is a
   valid, publishable result.

## Repo layout
matrix.yaml              # models, quants, files, hashes, gen config, pipeline_version
tasks/generate_tasks.py  # schema-driven task generation → tasks/tasks.jsonl
tasks/validators.py      # one validator class per task family
runner/run.py            # loops matrix × tasks × seeds, writes results/*.json
scorer/score.py          # reads results/, writes scores/scores.parquet
scorer/aggregate.py      # paired bootstrap → summary/summary.json (with CIs)
report/make_charts.py    # headline + per-metric charts (matplotlib, no seaborn deps)
space/                   # static HF Space (plain HTML/JS reading summary.json)
notebooks/kaggle_run_{model}.ipynb  # thin wrappers calling runner/run.py
future-work.md           # scope graveyard — everything out-of-scope goes here

## Environment
- Local dev: CPU only, tiny smoke model (Qwen2.5-0.5B GGUF) for end-to-end tests.
- Real runs: Kaggle T4 via notebooks/. llama-cpp-python pinned version in
  requirements.txt (pin it on day 1; llama.cpp moves fast).
- Tests: pytest. Every validator has unit tests with known-good and known-bad
  outputs. The runner has a dry-run mode (--limit 5 --model smoke).

## Definition of done
manifest.py reports 30,000/30,000 result files → scores + CIs computed →
charts render → Space deployed → dataset uploaded with raw outputs →
write-up drafted → one cell reproduced from scratch in a fresh Kaggle session.
Anything beyond this list is future-work.md material.

## Scope deflection
If the user asks for something outside frozen scope, append it to future-work.md
and say so, instead of implementing it.
