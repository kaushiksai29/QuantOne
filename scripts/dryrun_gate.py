"""Phase 3 dry-run gate. Given a scored v2 dry-run (one model, all its
quants), checks the ⛔ acceptance criteria before bulk spend:

  G1  FP16 (or the model's baseline) json_parse_rate > 0.50
      -> below this = chat-template bug, STOP.
  G2  decline subset (correct_decline_rate) paired-bootstrap 95% CI half-width
      <= 0.08 at every quant vs baseline -> the n=30->200 power fix worked.
  G3  zero think-leaks anywhere -> the /no_think enforcement held.

Reuses the vendored bootstrap from scorer/aggregate.py.

  python scripts/dryrun_gate.py --scores scores_v2.parquet --model qwen35_4b
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorer.aggregate import bootstrap_delta  # vendored, unchanged

QUANT_ORDER = ["FP16", "Q8_0", "Q4_K_M", "IQ4_XS", "Q3_K_M", "IQ3_M"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="scores_v2.parquet")
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline", default=None,
                    help="default: FP16 if present else Q8_0")
    ap.add_argument("--ci-max", type=float, default=0.08)
    args = ap.parse_args()

    df = pd.read_parquet(args.scores)
    df = df[(df.model == args.model) & (df.kv_type == "f16")]
    if df.empty:
        sys.exit(f"no rows for {args.model}")
    quants = [q for q in QUANT_ORDER if q in set(df.quant)]
    baseline = args.baseline or ("FP16" if "FP16" in quants else "Q8_0")
    print(f"model={args.model}  baseline={baseline}  quants={quants}\n")

    passed = True

    # G1 — parse rate at baseline
    pr = df[(df.quant == baseline) & (df.metric == "json_parse_rate")]["value"].mean()
    g1 = pr > 0.50
    passed &= g1
    print(f"G1 parse rate @ {baseline}: {pr:.3f}  ->  {'PASS' if g1 else 'FAIL (template bug — STOP)'}")

    # G3 — think leaks
    leaks = int(df["think_leak"].sum())
    g3 = leaks == 0
    passed &= g3
    print(f"G3 think leaks: {leaks}  ->  {'PASS' if g3 else 'FAIL (/no_think not holding)'}")

    # G2 — decline CI half-width per quant vs baseline (paired bootstrap over prompts)
    dec = df[df.metric == "correct_decline_rate"]
    per_prompt = (dec.groupby(["quant", "prompt_id"], as_index=False)["value"]
                  .mean())
    wide = per_prompt.pivot(index="prompt_id", columns="quant", values="value")
    rng = np.random.default_rng(12345)
    print(f"\nG2 decline-rate CI vs {baseline} (n={wide[baseline].notna().sum()} "
          f"decline prompts, target half-width <= {args.ci_max}):")
    worst = 0.0
    for q in quants:
        if q == baseline:
            continue
        pair = wide[[baseline, q]].dropna()
        delta, lo, hi = bootstrap_delta(pair[baseline], pair[q], 10000, rng)
        half = (hi - lo) / 2
        worst = max(worst, half)
        print(f"  {q:8s} delta={delta:+.3f} CI=[{lo:+.3f},{hi:+.3f}] "
              f"half-width={half:.3f} {'ok' if half <= args.ci_max else 'WIDE'}")
    g2 = worst <= args.ci_max
    passed &= g2
    print(f"G2 worst half-width: {worst:.3f}  ->  {'PASS' if g2 else 'FAIL'}")

    print(f"\n{'=== DRY RUN PASSED — proceed to Phase 4 ===' if passed else '=== DRY RUN FAILED — do not bulk-spend ==='}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
