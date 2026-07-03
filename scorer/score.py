"""Reads results/*.json, applies each task's validator, writes
scores/scores.parquet — one row per (model, quant, prompt_id, seed,
metric, value).

Scoring reads files only; it NEVER triggers inference and never imports
the runner (CLAUDE.md rules 2, 6). Its only cross-stage dependency is
tasks/validators.py — the validator contract shipped with the tasks.

  python scorer/score.py [--results-dir results] [--tasks tasks/tasks.jsonl]
                         [--out scores/scores.parquet]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasks.validators import VALIDATORS

# validator metric -> canonical metric name in scores/summary
METRIC_NAMES = {
    "json_parses": "json_parse_rate",
    "schema_complies": "schema_compliance",
    "tool_selection_correct": "tool_selection_acc",
    "arguments_exact_match": "argument_exact_match",
    "correctly_declined": "correct_decline_rate",
}


def score_results(results_dir, tasks_path):
    tasks = {}
    with open(tasks_path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            tasks[t["prompt_id"]] = t

    rows = []
    files = sorted(Path(results_dir).glob("*.json"))
    for path in files:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        task = tasks.get(r["prompt_id"])
        if task is None:
            sys.exit(f"ABORT: result {path.name} references unknown "
                     f"prompt_id {r['prompt_id']} — wrong tasks.jsonl?")
        validator = VALIDATORS[task["validator_id"]]
        metrics = validator.validate(r["raw_output"], task)
        for name, value in metrics.items():
            rows.append({
                "model": r["model"],
                "quant": r["quant"],
                "prompt_id": r["prompt_id"],
                "family": task["family"],
                "seed": r["seed"],
                "pipeline_version": r["pipeline_version"],
                "metric": METRIC_NAMES[name],
                "value": float(value),
            })
    return pd.DataFrame(rows), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    ap.add_argument("--out", default="scores/scores.parquet")
    args = ap.parse_args()

    t0 = time.perf_counter()
    df, n_files = score_results(args.results_dir, args.tasks)
    if df.empty:
        sys.exit(f"ABORT: no result files in {args.results_dir}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    dt = time.perf_counter() - t0
    print(f"scored {n_files} result files -> {len(df)} score rows "
          f"-> {out} in {dt:.1f}s")


if __name__ == "__main__":
    main()
