"""v2 scoring: reads results_v2/*.json, applies the v1 validators (vendored,
unchanged), writes scores_v2.parquet with kv_type carried through. Also
records a think_leak flag per row so the dry-run gate can verify thinking
stayed off. Never triggers inference.

  python scorer/score_v2.py [--results-dir results_v2] [--out scores_v2.parquet]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorer.score import METRIC_NAMES        # vendored, unchanged
from tasks.validators import VALIDATORS      # vendored, unchanged

THINK_RE = re.compile(r"<think|<thinking|</think", re.IGNORECASE)


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
            sys.exit(f"ABORT: {path.name} references unknown {r['prompt_id']}")
        metrics = VALIDATORS[task["validator_id"]].validate(r["raw_output"], task)
        leak = bool(THINK_RE.search(r["raw_output"]))
        for name, value in metrics.items():
            rows.append({
                "model": r["model"], "quant": r["quant"],
                "kv_type": r.get("kv_type", "f16"),
                "prompt_id": r["prompt_id"], "family": task["family"],
                "seed": r["seed"], "pipeline_version": r["pipeline_version"],
                "metric": METRIC_NAMES[name], "value": float(value),
                "think_leak": leak})
    return pd.DataFrame(rows), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_v2")
    ap.add_argument("--tasks", default="tasks/tasks_v2.jsonl")
    ap.add_argument("--out", default="scores_v2.parquet")
    args = ap.parse_args()
    t0 = time.perf_counter()
    df, n = score_results(args.results_dir, args.tasks)
    if df.empty:
        sys.exit(f"ABORT: no result files in {args.results_dir}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"scored {n} files -> {len(df)} rows -> {args.out} "
          f"in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
