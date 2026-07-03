"""Aggregates scores/scores.parquet -> summary/summary.json.

Per (model, quant, metric): mean across seeds then prompts. For every
quant-vs-baseline (FP16) delta per model per metric: paired bootstrap —
seeds are averaged first, then 10,000 resamples over prompt_ids — giving
a 95% CI. significant=true only when the CI excludes zero (CLAUDE.md
rule 8). The bootstrap RNG seed is fixed so summary.json is reproducible.

  python scorer/aggregate.py [--scores scores/scores.parquet]
                             [--out summary/summary.json]
                             [--baseline FP16] [--n-boot 10000]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_SEED = 12345


def seed_averaged(df):
    """One value per (model, quant, metric, prompt_id): mean over seeds."""
    return (df.groupby(["model", "quant", "metric", "prompt_id"],
                       as_index=False)["value"].mean())


def bootstrap_delta(baseline_vals, quant_vals, n_boot, rng):
    """Paired bootstrap over prompts. Inputs are aligned per-prompt
    (seed-averaged) arrays. Returns (delta, ci_lo, ci_hi)."""
    diffs = np.asarray(quant_vals, dtype=float) - np.asarray(baseline_vals,
                                                             dtype=float)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def compute_summary(df, baseline="FP16", n_boot=10_000, seed=BOOTSTRAP_SEED):
    per_prompt = seed_averaged(df)
    rng = np.random.default_rng(seed)

    cells = []
    for (model, quant, metric), g in per_prompt.groupby(
            ["model", "quant", "metric"]):
        cells.append({"model": model, "quant": quant, "metric": metric,
                      "mean": float(g["value"].mean()),
                      "n_prompts": int(g["prompt_id"].nunique())})

    deltas = []
    for (model, metric), g in per_prompt.groupby(["model", "metric"]):
        wide = g.pivot(index="prompt_id", columns="quant", values="value")
        if baseline not in wide.columns:
            continue  # no baseline results for this model yet
        for quant in wide.columns:
            if quant == baseline:
                continue
            pair = wide[[baseline, quant]].dropna()
            delta, lo, hi = bootstrap_delta(pair[baseline], pair[quant],
                                            n_boot, rng)
            deltas.append({
                "model": model, "metric": metric, "quant": quant,
                "baseline": baseline, "delta": delta,
                "ci_lo": lo, "ci_hi": hi,
                "n_prompts": int(len(pair)),
                "significant": bool(lo > 0 or hi < 0),
            })

    return {
        "meta": {
            "baseline": baseline,
            "n_boot": n_boot,
            "bootstrap_seed": seed,
            "pipeline_version": int(df["pipeline_version"].iloc[0])
            if "pipeline_version" in df else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "cells": cells,
        "deltas": deltas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="scores/scores.parquet")
    ap.add_argument("--out", default="summary/summary.json")
    ap.add_argument("--baseline", default="FP16")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    df = pd.read_parquet(args.scores)
    summary = compute_summary(df, baseline=args.baseline, n_boot=args.n_boot)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    n_sig = sum(d["significant"] for d in summary["deltas"])
    print(f"{len(summary['cells'])} cells, {len(summary['deltas'])} deltas "
          f"({n_sig} significant) -> {out}")


if __name__ == "__main__":
    main()
