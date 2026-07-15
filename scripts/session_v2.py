"""One v2 session for ONE model: ingest prior results, then for each quant
in the manifest download the GGUF (split-file aware), verify sha256, run
runner/run_v2.py, delete the GGUF (disk budget), and zip results_v2/ for
cross-session accumulation. Optionally includes the KV arm and QAT arm for
models that have them.

  python scripts/session_v2.py --model qwen35_4b --quants FP16,Q8_0,Q4_K_M \
      --results-dir results_v2 --zip-to results_v2_qwen35_4b.zip
"""

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common_v2 import (DEFAULT_KV, KV_ARM_MODELS, KV_ARM_TYPES,
                       KV_ARM_WEIGHT_QUANT, REPO_ROOT, load_manifest,
                       sha256_file)


def ingest(sources, results_dir):
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
                    if name.endswith(".json") and not (results_dir / name).exists():
                        with zf.open(info) as fin, open(results_dir / name, "wb") as fo:
                            shutil.copyfileobj(fin, fo)
                        copied += 1
        else:
            for p in src.glob("*.json"):
                if not (results_dir / p.name).exists():
                    shutil.copy2(p, results_dir / p.name)
                    copied += 1
    print(f"ingested {copied} prior result files")


def download_verify(entry, quant, repo, models_dir):
    finfo = entry["files"][quant]
    parts = finfo if isinstance(finfo, list) else [finfo]
    models_dir.mkdir(parents=True, exist_ok=True)
    local = []
    for p in parts:
        path = models_dir / Path(p["path"]).name
        url = f"https://huggingface.co/{repo}/resolve/main/{p['path']}"
        if not path.exists():
            print(f"  downloading {p['path']} ({p['gb']}GB) ...")
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.replace(path)
        if sha256_file(path) != p["sha256"]:
            sys.exit(f"ABORT: sha256 mismatch for {path} (rule 5).")
        local.append(path)
    print(f"  verified {quant} sha256 OK")
    return local


def run_quant(model, quant, kv, args):
    cmd = [sys.executable, str(REPO_ROOT / "runner/run_v2.py"),
           "--model", model, "--quant", quant, "--kv-type", kv,
           "--tasks", args.tasks, "--results-dir", args.results_dir,
           "--models-dir", args.models_dir, "--manifest", args.manifest,
           "--n-gpu-layers", str(args.n_gpu_layers)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quants", default=None, help="comma list; default = all in manifest")
    ap.add_argument("--tasks", default=str(REPO_ROOT / "tasks/tasks_v2.jsonl"))
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results_v2"))
    ap.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    ap.add_argument("--manifest", default=str(REPO_ROOT / "manifest.json"))
    ap.add_argument("--prior-results", action="append", default=[])
    ap.add_argument("--zip-to", default=None)
    ap.add_argument("--keep-ggufs", action="store_true")
    ap.add_argument("--with-arms", action="store_true",
                    help="also run this model's KV/QAT arm conditions")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    entry = manifest["models"][args.model]
    repo = entry["gguf_repo"]
    quants = args.quants.split(",") if args.quants else list(entry["files"])
    results_dir = Path(args.results_dir)

    if args.prior_results:
        ingest(args.prior_results, results_dir)

    for quant in quants:
        gguf = download_verify(entry, quant, repo, Path(args.models_dir))[0]
        run_quant(args.model, quant, DEFAULT_KV, args)
        # KV arm: same Q4_K_M weights, extra cache types (no extra download)
        if args.with_arms and args.model in KV_ARM_MODELS and quant == KV_ARM_WEIGHT_QUANT:
            for kv in KV_ARM_TYPES:
                run_quant(args.model, quant, kv, args)
        if not args.keep_ggufs:
            gguf.unlink()
            print(f"  deleted {gguf.name} (disk budget)")

    # QAT arm: separate official q4_0 GGUF
    if args.with_arms and "qat" in entry:
        q = entry["qat"]
        path = Path(args.models_dir) / Path(q["path"]).name
        url = f"https://huggingface.co/{q['repo']}/resolve/main/{q['path']}"
        if not path.exists():
            print(f"  downloading QAT {q['path']} ({q['gb']}GB) ...")
            urllib.request.urlretrieve(url, path)
        if sha256_file(path) != q["sha256"]:
            sys.exit("ABORT: QAT sha256 mismatch (rule 5).")
        run_quant(args.model, "QAT_q4_0", DEFAULT_KV, args)
        if not args.keep_ggufs:
            path.unlink()

    if args.zip_to:
        zp = Path(args.zip_to)
        zp.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(results_dir.glob("*.json")):
                zf.write(p, p.name)
        print(f"zipped {len(zf.namelist())} results -> {zp}")

    subprocess.run([sys.executable, str(REPO_ROOT / "manifest_v2.py"),
                    "--model", args.model, "--tasks", args.tasks,
                    "--results-dir", args.results_dir,
                    "--manifest", args.manifest], check=True)


if __name__ == "__main__":
    main()
