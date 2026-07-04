"""Fresh-environment reproduction of ONE (model, quant) cell: regenerates
all 1,500 generations from scratch into a temporary results dir, scores
them, and asserts each metric's mean matches the published summary.json
within tolerance.

Exact equality is NOT expected: greedy decoding is not bit-stable across
hardware/thread counts (see WRITEUP.md methods note), so per-metric means
are compared within --tol (default 0.03, roughly the observed replicate
noise). Requires the cell's GGUF downloaded (scripts/kaggle_session.py
does download+verify) — or run inside a Kaggle/Colab session.

  python scripts/repro_cell.py --model qwen25_3b --quant Q3_K_M \
      [--n-gpu-layers -1] [--tol 0.03]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--summary", default=str(REPO_ROOT / "summary/summary.json"))
    ap.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.03)
    args = ap.parse_args()

    with open(args.summary, encoding="utf-8") as f:
        published = {(c["quant"], c["metric"]): c["mean"]
                     for c in json.load(f)["cells"]
                     if c["model"] == args.model}
    if not published:
        sys.exit(f"ABORT: no published cells for {args.model} in {args.summary}")

    tmp = Path(tempfile.mkdtemp(prefix="quantone_repro_"))
    results, scores = tmp / "results", tmp / "scores.parquet"
    print(f"repro dir: {tmp}")

    subprocess.run([sys.executable, str(REPO_ROOT / "runner/run.py"),
                    "--model", args.model, "--quant", args.quant,
                    "--seeds", "3", "--tasks", str(REPO_ROOT / "tasks/tasks.jsonl"),
                    "--results-dir", str(results),
                    "--models-dir", args.models_dir,
                    "--n-gpu-layers", str(args.n_gpu_layers)], check=True)
    subprocess.run([sys.executable, str(REPO_ROOT / "scorer/score.py"),
                    "--results-dir", str(results),
                    "--tasks", str(REPO_ROOT / "tasks/tasks.jsonl"),
                    "--out", str(scores)], check=True)

    import pandas as pd
    df = pd.read_parquet(scores)
    per_prompt = df.groupby(["metric", "prompt_id"])["value"].mean()
    repro_means = per_prompt.groupby("metric").mean()

    failures = []
    for metric, mean in repro_means.items():
        pub = published.get((args.quant, metric))
        diff = abs(mean - pub)
        status = "OK " if diff <= args.tol else "FAIL"
        print(f"{status} {metric:22s} repro={mean:.3f} published={pub:.3f} "
              f"|diff|={diff:.3f}")
        if diff > args.tol:
            failures.append(metric)
    if failures:
        sys.exit(f"REPRO FAILED for {failures} (tol={args.tol})")
    print(f"REPRO PASSED: {args.model}/{args.quant} matches published "
          f"summary within {args.tol}")


if __name__ == "__main__":
    main()
