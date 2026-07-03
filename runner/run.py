"""Content-addressed, resumable runner over matrix x tasks x seeds.

CLAUDE.md rules 2, 3, 7:
  - Every generation writes results/{sha1(model|quant|prompt_id|seed)}.json.
    Existing files are skipped, so re-running resumes for free and is
    always idempotent (crash-safe: write tmp, then atomic rename).
  - Generation config is read from matrix.yaml only (frozen).
  - GGUF sha256 is verified against matrix.yaml before loading; any
    mismatch or unrecorded hash aborts the run (rule 5).

Usage:
  python runner/run.py --model smoke --quant Q8_0 --seeds 1 --limit 20 \
      --tasks tasks/tasks.jsonl
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_matrix, load_tasks, model_config, result_key, sha256_file


def verify_gguf(cfg, quant, models_dir):
    entry = cfg["files"][quant]
    path = Path(models_dir) / entry["gguf_url"].rsplit("/", 1)[-1]
    if not path.exists():
        sys.exit(f"ABORT: model file not found: {path}\n"
                 f"Download it from {entry['gguf_url']} first.")
    expected = entry["sha256"]
    if not expected or expected == "TODO":
        sys.exit(f"ABORT: no sha256 recorded in matrix.yaml for {quant} "
                 f"(rule 5: record the hash before the first run).")
    actual = sha256_file(path)
    if actual != expected:
        sys.exit(f"ABORT: sha256 mismatch for {path}\n"
                 f"  expected {expected}\n  actual   {actual}\n"
                 f"Do not silently redownload (rule 5).")
    return path


def write_atomic(path, record):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, sort_keys=True, ensure_ascii=True)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model key in matrix.yaml (or 'smoke')")
    ap.add_argument("--quant", required=True)
    ap.add_argument("--seeds", type=int, default=3, help="runs seeds 0..N-1")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N tasks")
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=0,
                    help="layers to offload to GPU (-1 = all); execution "
                         "detail, not part of the frozen gen config")
    args = ap.parse_args()

    matrix = load_matrix(args.matrix)
    cfg = model_config(matrix, args.model)
    gen_config = matrix["gen_config"]
    pipeline_version = matrix["pipeline_version"]

    gguf_path = verify_gguf(cfg, args.quant, args.models_dir)
    tasks = load_tasks(args.tasks, args.limit)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for task in tasks:
        for seed in range(args.seeds):
            key = result_key(args.model, args.quant, task["prompt_id"], seed)
            if not (results_dir / f"{key}.json").exists():
                pending.append((task, seed, key))
    total = len(tasks) * args.seeds
    print(f"[{args.model}/{args.quant}] {total - len(pending)}/{total} done, "
          f"{len(pending)} pending")
    if not pending:
        return

    from llama_cpp import Llama  # imported late: manifest/scoring never need it

    n_ctx = min(args.n_ctx, int(cfg["context_length"]))
    llm = Llama(model_path=str(gguf_path), n_ctx=n_ctx,
                chat_format=cfg["chat_template"],
                n_gpu_layers=args.n_gpu_layers, verbose=False)

    for i, (task, seed, key) in enumerate(pending, 1):
        t0 = time.perf_counter()
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": task["prompt"]}],
            temperature=gen_config["temperature"],
            max_tokens=gen_config["max_tokens"],
            seed=seed,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        write_atomic(results_dir / f"{key}.json", {
            "result_key": key,
            "model": args.model,
            "model_id": cfg["model_id"],
            "quant": args.quant,
            "prompt_id": task["prompt_id"],
            "family": task["family"],
            "validator_id": task["validator_id"],
            "seed": seed,
            "raw_output": out["choices"][0]["message"]["content"],
            "finish_reason": out["choices"][0]["finish_reason"],
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": pipeline_version,
        })
        if i % 10 == 0 or i == len(pending):
            print(f"  {i}/{len(pending)} ({task['prompt_id']} seed={seed}, "
                  f"{latency_ms} ms)")


if __name__ == "__main__":
    main()
