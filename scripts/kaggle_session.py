"""One Kaggle session for ONE model: ingest prior results, then for each
quant download the GGUF, verify sha256 against matrix.yaml (abort on
mismatch — rule 5), run runner/run.py, delete the GGUF to stay inside
Kaggle's disk budget, and finally zip results/ for cross-session
accumulation.

The notebook (notebooks/kaggle_run_template.ipynb) is a thin wrapper
around this script; running it locally with --model smoke is the Phase 3
dry-run.

  python scripts/kaggle_session.py --model smoke --quants Q8_0,Q4_K_M \
      --seeds 1 --limit 20 --keep-ggufs --zip-to results_smoke.zip
"""

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import REPO_ROOT, load_matrix, model_config, sha256_file


def ingest_prior_results(sources, results_dir):
    """Copy result JSONs from prior zips/dirs into results_dir. Existing
    files are never overwritten (content-addressed names make any
    collision an identical record anyway)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sources:
        src = Path(src)
        if not src.exists():
            sys.exit(f"ABORT: prior-results source not found: {src}")
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as zf:
                for info in zf.infolist():
                    name = Path(info.filename).name
                    if not name.endswith(".json"):
                        continue
                    dest = results_dir / name
                    if not dest.exists():
                        with zf.open(info) as fin, open(dest, "wb") as fout:
                            shutil.copyfileobj(fin, fout)
                        copied += 1
        else:
            for p in src.glob("*.json"):
                dest = results_dir / p.name
                if not dest.exists():
                    shutil.copy2(p, dest)
                    copied += 1
    print(f"ingested {copied} prior result files into {results_dir}")


def download_and_verify(cfg, quant, models_dir):
    """Download the GGUF if missing, then verify sha256. A mismatch always
    aborts and never deletes/redownloads (rule 5)."""
    entry = cfg["files"][quant]
    url, expected = entry["gguf_url"], entry["sha256"]
    if not expected or expected == "TODO":
        sys.exit(f"ABORT: no sha256 recorded in matrix.yaml for {quant} "
                 f"(rule 5: record the hash before the first run).")
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / url.rsplit("/", 1)[-1]
    if not path.exists():
        print(f"downloading {url} ...")
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        tmp.replace(path)
    actual = sha256_file(path)
    if actual != expected:
        sys.exit(f"ABORT: sha256 mismatch for {path}\n"
                 f"  expected {expected}\n  actual   {actual}\n"
                 f"Do not silently redownload (rule 5).")
    print(f"verified {path.name} sha256 OK")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quants", default=None,
                    help="comma-separated; default: matrix quants present in "
                         "the model's files")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", default=str(REPO_ROOT / "tasks/tasks.jsonl"))
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    ap.add_argument("--prior-results", action="append", default=[],
                    help="zip file or directory of prior result JSONs; repeatable")
    ap.add_argument("--zip-to", default=None,
                    help="write a zip of results/ here when the session ends")
    ap.add_argument("--keep-ggufs", action="store_true",
                    help="do not delete GGUFs after each quant (local dev)")
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    args = ap.parse_args()

    matrix = load_matrix()
    cfg = model_config(matrix, args.model)
    quants = (args.quants.split(",") if args.quants
              else [q for q in matrix["quants"] if q in cfg["files"]])
    seeds = args.seeds if args.seeds is not None else matrix["seeds"]
    results_dir = Path(args.results_dir)

    if args.prior_results:
        ingest_prior_results(args.prior_results, results_dir)

    for quant in quants:
        gguf_path = download_and_verify(cfg, quant, Path(args.models_dir))
        cmd = [sys.executable, str(REPO_ROOT / "runner/run.py"),
               "--model", args.model, "--quant", quant,
               "--seeds", str(seeds), "--tasks", args.tasks,
               "--results-dir", str(results_dir),
               "--models-dir", args.models_dir,
               "--n-gpu-layers", str(args.n_gpu_layers)]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        subprocess.run(cmd, check=True)
        if not args.keep_ggufs:
            gguf_path.unlink()
            print(f"deleted {gguf_path.name} (disk budget)")

    if args.zip_to:
        zip_path = Path(args.zip_to)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(results_dir.glob("*.json")):
                zf.write(p, p.name)
        print(f"zipped {len(zf.namelist())} result files -> {zip_path}")

    manifest_cmd = [sys.executable, str(REPO_ROOT / "manifest.py"),
                    "--model", args.model, "--quants", ",".join(quants),
                    "--seeds", str(seeds), "--tasks", args.tasks,
                    "--results-dir", str(results_dir)]
    if args.limit is not None:
        manifest_cmd += ["--limit", str(args.limit)]
    subprocess.run(manifest_cmd, check=True)


if __name__ == "__main__":
    main()
