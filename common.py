"""Shared plumbing for the runner stage (run.py + manifest.py): the
content-addressing scheme (CLAUDE.md rule 2) and matrix.yaml loading.
Scoring stages never import this — they read fields from result files."""

import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent


def result_key(model_key, quant, prompt_id, seed):
    """sha1(model + quant + prompt_id + seed) — the result filename stem."""
    raw = f"{model_key}|{quant}|{prompt_id}|{seed}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_matrix(path=None):
    with open(path or REPO_ROOT / "matrix.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_config(matrix, model_key):
    if model_key == "smoke":
        return matrix["smoke"]
    return matrix["models"][model_key]


def load_tasks(path, limit=None):
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
