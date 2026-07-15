"""v2 progress manifest: expected vs done cell counts per (model, quant,
kv_type), honoring the thorough-v2 knobs. Never runs inference.

  python manifest_v2.py                      # whole study
  python manifest_v2.py --model qwen35_2b    # one model
"""

import argparse
from pathlib import Path

from common_v2 import (DEFAULT_KV, KV_ARM_MODELS, KV_ARM_TYPES,
                       KV_ARM_WEIGHT_QUANT, cell_key, load_manifest,
                       load_tasks, plan_cells)


def cells_for(model, quant, kv_type, tasks):
    return list(plan_cells(model, quant, kv_type, tasks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tier", default=None, choices=["small", "big"])
    ap.add_argument("--tasks", default="tasks/tasks_v2.jsonl")
    ap.add_argument("--results-dir", default="results_v2")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    tasks = load_tasks(args.tasks)
    existing = {p.stem for p in Path(args.results_dir).glob("*.json")}

    grand_done = grand_exp = 0
    for model, entry in sorted(manifest["models"].items()):
        if args.model and model != args.model:
            continue
        if args.tier and entry["tier"] != args.tier:
            continue
        # normal cells: every quant at kv f16
        conditions = [(q, DEFAULT_KV) for q in entry["files"]]
        # KV arm: extra kv types at the weight quant
        if model in KV_ARM_MODELS and KV_ARM_WEIGHT_QUANT in entry["files"]:
            conditions += [(KV_ARM_WEIGHT_QUANT, kv) for kv in KV_ARM_TYPES]
        # QAT arm
        if "qat" in entry:
            conditions.append(("QAT_q4_0", DEFAULT_KV))
        for quant, kv in conditions:
            if quant == "QAT_q4_0":
                planned = cells_for(model, "Q4_K_M", kv, tasks)  # QAT = full ladder task set
            else:
                planned = cells_for(model, quant, kv, tasks)
            done = sum(1 for (t, s) in planned
                       if cell_key(model, quant, kv, t["prompt_id"], s) in existing)
            exp = len(planned)
            grand_done += done
            grand_exp += exp
            mark = "OK " if done == exp else "   "
            tag = f"{quant}/kv={kv}" if kv != DEFAULT_KV else quant
            print(f"{mark}{model:16s} {tag:18s} {done:6d}/{exp}")
    print(f"\nTOTAL {grand_done}/{grand_exp}")
    return 0 if grand_done == grand_exp else 1


if __name__ == "__main__":
    raise SystemExit(main())
