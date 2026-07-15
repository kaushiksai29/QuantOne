"""QuantOne v2 runner. Manifest-driven; content-addressed and resumable
(CLAUDE.md rules 2, 3, 7) with v2 additions:
  - per-model thinking-off enforcement (from manifest thinking_off_mechanism),
  - KV cache-type flags (--kv-type, threaded into llama.cpp via cache_type_k/v),
  - cell-level SHA-256 of the raw output recorded in each result,
  - a structured JSONL run log (one line per generated cell),
  - thorough-v2 knobs applied via common_v2.plan_cells.

Every result: results/{sha1(model|quant|kv_type|prompt_id|seed)}.json.
Existing files are skipped; writes are tmp-then-atomic-rename.

  python runner/run_v2.py --model qwen35_2b --quant Q4_K_M \
      --tasks tasks/tasks_v2.jsonl --models-dir models
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common_v2 import (DEFAULT_KV, cell_key, load_manifest, load_matrix_v2,
                       load_tasks, plan_cells, sha256_file,
                       thinking_off_messages)


def resolve_file(entry, quant, models_dir):
    """Return the local path for a quant's GGUF, verifying sha256. Handles
    single-file and split (multi-part) entries; aborts on mismatch (rule 5)."""
    finfo = entry["files"][quant]
    parts = finfo if isinstance(finfo, list) else [finfo]
    local_parts = []
    for p in parts:
        path = Path(models_dir) / Path(p["path"]).name
        if not path.exists():
            sys.exit(f"ABORT: model file not found: {path}")
        if sha256_file(path) != p["sha256"]:
            sys.exit(f"ABORT: sha256 mismatch for {path} (rule 5).")
        local_parts.append(path)
    return local_parts[0]  # llama.cpp opens the first shard of a split set


def write_atomic(path, record):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, sort_keys=True, ensure_ascii=True)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--kv-type", default=DEFAULT_KV,
                    choices=["f16", "q8_0", "q4_0"])
    ap.add_argument("--tasks", default="tasks/tasks_v2.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results-dir", default="results_v2")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--run-log", default=None, help="JSONL run log path")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    matrix = load_matrix_v2(args.matrix)
    entry = manifest["models"][args.model]
    if args.quant not in entry["files"]:
        sys.exit(f"ABORT: {args.model} has no {args.quant} in the manifest "
                 f"(available: {list(entry['files'])}).")
    gen_config = matrix["gen_config"]
    pipeline_version = matrix["pipeline_version"]
    mechanism = entry["thinking_off_mechanism"]

    gguf_path = resolve_file(entry, args.quant, args.models_dir)
    tasks = load_tasks(args.tasks, args.limit)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_log = Path(args.run_log) if args.run_log else \
        results_dir / f"runlog_{args.model}_{args.quant}_{args.kv_type}.jsonl"

    planned = list(plan_cells(args.model, args.quant, args.kv_type, tasks))
    pending = [(t, s) for (t, s) in planned
               if not (results_dir / f"{cell_key(args.model, args.quant, args.kv_type, t['prompt_id'], s)}.json").exists()]
    print(f"[{args.model}/{args.quant}/kv={args.kv_type}] "
          f"{len(planned) - len(pending)}/{len(planned)} done, "
          f"{len(pending)} pending")
    if not pending:
        return

    from llama_cpp import Llama

    n_ctx = min(args.n_ctx, int(entry.get("context_length") or args.n_ctx))
    kv_kwargs = {}
    if args.kv_type != "f16":
        # q4_0/q8_0 KV cache typically needs flash attention on in llama.cpp
        kv_kwargs = {"type_k": args.kv_type, "type_v": args.kv_type,
                     "flash_attn": True}
    llm = Llama(model_path=str(gguf_path), n_ctx=n_ctx, chat_format=None,
                n_gpu_layers=args.n_gpu_layers, verbose=False, **kv_kwargs)

    log = open(run_log, "a", encoding="utf-8", newline="\n")
    for i, (task, seed) in enumerate(pending, 1):
        key = cell_key(args.model, args.quant, args.kv_type,
                       task["prompt_id"], seed)
        messages = thinking_off_messages(mechanism, task["prompt"])
        t0 = time.perf_counter()
        out = llm.create_chat_completion(
            messages=messages, temperature=gen_config["temperature"],
            max_tokens=gen_config["max_tokens"], seed=seed)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        raw = out["choices"][0]["message"]["content"]
        record = {
            "cell_key": key, "model": args.model, "model_id": entry["base_model"],
            "quant": args.quant, "kv_type": args.kv_type,
            "thinking_off_mechanism": mechanism,
            "prompt_id": task["prompt_id"], "family": task["family"],
            "validator_id": task["validator_id"], "seed": seed,
            "raw_output": raw,
            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "finish_reason": out["choices"][0]["finish_reason"],
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": pipeline_version,
        }
        write_atomic(results_dir / f"{key}.json", record)
        log.write(json.dumps({"key": key, "prompt_id": task["prompt_id"],
                              "seed": seed, "raw_sha256": record["raw_sha256"],
                              "finish_reason": record["finish_reason"],
                              "latency_ms": latency_ms}) + "\n")
        log.flush()
        if i % 50 == 0 or i == len(pending):
            print(f"  {i}/{len(pending)} ({task['prompt_id']} seed={seed}, "
                  f"{latency_ms} ms)")
    log.close()


if __name__ == "__main__":
    main()
