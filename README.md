# QuantOne — quantization vs structured-output reliability

Does quantization (FP16 → Q8 → Q4_K_M → Q3_K_M) hurt an LLM's ability to
produce valid structured output? 5 models × 4 quants × 500 machine-checkable
tasks × 3 seeds = 30,000 generations, scored deterministically (no LLM
judges), aggregated with paired bootstrap 95% CIs. See `CLAUDE.md` for the
eight non-negotiable rules; anything out of scope lives in `future-work.md`.

## Pipeline

```
tasks/generate_tasks.py  -> tasks/tasks.jsonl        (500 tasks, seed-fixed)
runner/run.py            -> results/{sha1}.json      (content-addressed, resumable)
scorer/score.py          -> scores/scores.parquet
scorer/aggregate.py      -> summary/summary.json     (paired bootstrap CIs)
report/make_charts.py    -> charts
space/                   -> static leaderboard reading summary.json
```

Each stage only reads the previous stage's files; none imports another's
internals. `python manifest.py` reports result-file counts vs the expected
30,000 at any time.

## Local dev (CPU)

```bash
pip install -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pytest                      # validators, generator, bootstrap tests
python tasks/generate_tasks.py
make smoke                  # 0.5B model x 2 quants x 20 prompts on CPU
```

(No `make` on Windows: run the commands from the Makefile target directly.)

## Running the real matrix on Kaggle

One notebook per model, cloned from
`notebooks/kaggle_run_template.ipynb` with `MODEL_KEY` set:

1. **Accelerator.** GPU T4 for 7–8B models. 1–4B models run fine as **CPU
   sessions**, and several CPU notebooks can run in parallel alongside a GPU
   session — that's how the whole matrix fits in free-tier hours.
2. **Internet on.** GGUFs are downloaded one quant at a time, sha256-verified
   against `matrix.yaml` (mismatch aborts — rule 5), run, then deleted to fit
   Kaggle's disk budget.
3. **Cross-session accumulation.** The session ends by zipping `results/` to
   `/kaggle/working/results_<model>.zip`. Save Version, then attach that
   output as an *input* of the next session: it is ingested at startup and the
   content-addressed runner (rule 2) skips everything already done. Repeat
   until the manifest cell prints complete counts. A ~12h session cap is
   therefore harmless (rule 7).
4. When all models report complete, download the zips, merge into one
   `results/` dir locally, and proceed to scoring.

Filling `matrix.yaml`: record `gguf_url` **and** `sha256` for every quant of
every model before its first run. Note the smoke entry as the format example.
