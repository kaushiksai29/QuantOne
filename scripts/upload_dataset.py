"""Uploads the QuantOne artifacts to a Hugging Face dataset repo:
tasks.jsonl, all 30k raw outputs (consolidated to results.jsonl.gz),
scores.parquet, summary.json, and a dataset card.

Requires `huggingface_hub` and an HF token with write access
(`huggingface-cli login` or HF_TOKEN env var). Run manually — publishing
is deliberate, not automatic.

  python scripts/upload_dataset.py --repo <user>/quantone [--private]
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import REPO_ROOT

CARD = """---
license: mit
tags: [llm, quantization, structured-output, tool-calling, gguf, benchmark]
---

# QuantOne — quantization vs structured-output reliability

30,000 generations: 5 models (1.7B-3.8B) x 4 quant levels (FP16/Q8_0/Q4_K_M/
Q3_K_M) x 500 machine-checkable structured-output & tool-call tasks x 3 seeds,
run with llama.cpp on free-tier T4s, scored deterministically (no LLM judges),
aggregated with paired bootstrap 95% CIs.

Headline: Q8_0 showed zero significant regressions across 75 comparisons;
Q3_K_M significantly degrades schema compliance in 3/5 models and collapses
should-not-call behavior in two model families.

Files: `tasks.jsonl` (500 tasks + gold + schemas), `results.jsonl.gz`
(30,000 raw generations with latency + finish_reason), `scores.parquet`
(67,200 metric rows), `summary.json` (means, deltas, CIs).

Code, method, and write-up: https://github.com/kaushiksai29/QuantOne
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. username/quantone")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    from huggingface_hub import HfApi

    results = sorted(Path(args.results_dir).glob("*.json"))
    if len(results) != 30_000:
        sys.exit(f"ABORT: expected 30000 result files, found {len(results)} — "
                 f"run manifest.py first.")

    consolidated = REPO_ROOT / "results.jsonl.gz"
    print(f"consolidating {len(results)} results -> {consolidated} ...")
    with gzip.open(consolidated, "wt", encoding="utf-8", newline="\n") as f:
        for p in results:
            with open(p, encoding="utf-8") as fin:
                f.write(json.dumps(json.load(fin), sort_keys=True,
                                   ensure_ascii=True) + "\n")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private,
                    exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    for local, remote in [
        (REPO_ROOT / "tasks/tasks.jsonl", "tasks.jsonl"),
        (consolidated, "results.jsonl.gz"),
        (REPO_ROOT / "scores/scores.parquet", "scores.parquet"),
        (REPO_ROOT / "summary/summary.json", "summary.json"),
        (REPO_ROOT / "matrix.yaml", "matrix.yaml"),
    ]:
        print(f"uploading {remote} ...")
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=args.repo, repo_type="dataset")
    consolidated.unlink()
    print(f"done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
