"""Reports per-(model, quant) result-file counts vs expected, and lists
missing keys. Reads matrix.yaml + tasks.jsonl + the results dir; never
runs inference.

Full matrix:      python manifest.py
Scoped (smoke):   python manifest.py --model smoke --quants Q8_0,Q4_K_M \
                      --seeds 1 --limit 20
"""

import argparse
from pathlib import Path

from common import load_matrix, load_tasks, result_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="restrict to one model key")
    ap.add_argument("--quants", default=None, help="comma-separated quant keys")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", default="tasks/tasks.jsonl")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--show-missing", type=int, default=10,
                    help="max missing keys to list per cell")
    args = ap.parse_args()

    matrix = load_matrix(args.matrix)
    models = [args.model] if args.model else sorted(matrix["models"])
    quants = args.quants.split(",") if args.quants else matrix["quants"]
    seeds = args.seeds if args.seeds is not None else matrix["seeds"]
    tasks = load_tasks(args.tasks, args.limit)
    existing = {p.stem for p in Path(args.results_dir).glob("*.json")}

    grand_done = grand_expected = 0
    for model in models:
        for quant in quants:
            missing = []
            done = 0
            for task in tasks:
                for seed in range(seeds):
                    key = result_key(model, quant, task["prompt_id"], seed)
                    if key in existing:
                        done += 1
                    else:
                        missing.append((task["prompt_id"], seed))
            expected = len(tasks) * seeds
            grand_done += done
            grand_expected += expected
            mark = "OK " if done == expected else "   "
            print(f"{mark}{model:12s} {quant:8s} {done:6d}/{expected}")
            for prompt_id, seed in missing[:args.show_missing]:
                print(f"      missing: {prompt_id} seed={seed}")
            if len(missing) > args.show_missing:
                print(f"      ... and {len(missing) - args.show_missing} more")
    print(f"\nTOTAL {grand_done}/{grand_expected}")
    return 0 if grand_done == grand_expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
