"""Shared plumbing for the v2 runner stage: content-addressing (now including
kv_type), the manifest, and the THOROUGH-v2 knob plan that decides which
(task, seed) cells actually run for a given (model, quant, kv_type).

Knob package (thorough v2, approved 2026-07-15):
  - seeds: 3 on tool_decline tasks (the power fix), 2 on structured_output
    and tool_call tasks.
  - IQ quants (IQ4_XS, IQ3_M) run on the v1-500 subset only; every other
    quant runs the full 1,200. K-quants/FP16/Q8 are never subset-restricted.
  - KV arm: kv_type in {q8_0, q4_0} only for the two KV-arm models at Q4_K_M
    weights; kv_type f16 is the default for all normal cells (so the KV arm's
    f16 condition == the main Q4_K_M run, reused, never regenerated).
Scoring stages never import this.
"""

import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent

IQ_QUANTS = {"IQ4_XS", "IQ3_M"}
DEFAULT_KV = "f16"
KV_ARM_MODELS = {"qwen35_4b", "gemma4_e4b"}
KV_ARM_WEIGHT_QUANT = "Q4_K_M"
KV_ARM_TYPES = ["q8_0", "q4_0"]           # f16 reuses the main Q4_K_M run
QAT_ARM_MODELS = {"gemma4_e4b", "gemma4_26b_a4b"}

SEEDS_BY_FAMILY = {"tool_decline": 3, "structured_output": 2, "tool_call": 2}


def is_v1_subset(prompt_id):
    """The 500 tasks shared byte-identically with v1 (see generate_tasks_v2)."""
    kind, num = prompt_id.rsplit("-", 1)
    n = int(num)
    return ((kind == "struct" and n < 350) or (kind == "tool" and n < 120)
            or (kind == "decline" and n < 30))


def cell_key(model_key, quant, kv_type, prompt_id, seed):
    raw = f"{model_key}|{quant}|{kv_type}|{prompt_id}|{seed}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def plan_cells(model_key, quant, kv_type, tasks):
    """Yield (task, seed) pairs to generate for this (model, quant, kv_type),
    applying the thorough-v2 knobs. Deterministic order."""
    iq_restricted = quant in IQ_QUANTS
    for task in tasks:
        if iq_restricted and not is_v1_subset(task["prompt_id"]):
            continue
        for seed in range(SEEDS_BY_FAMILY[task["family"]]):
            yield task, seed


def load_manifest(path=None):
    with open(path or REPO_ROOT / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def load_matrix_v2(path=None):
    """The frozen non-file config (gen params) lives in matrix_v2.yaml."""
    with open(path or REPO_ROOT / "matrix_v2.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tasks(path, limit=None):
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def thinking_off_messages(mechanism, prompt):
    """Apply the model's confirmed thinking-off mechanism to a task prompt."""
    if mechanism == "no_think_system":
        return [{"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt}]
    if mechanism in (None, "none", ""):
        return [{"role": "user", "content": prompt}]
    raise ValueError(f"unknown thinking_off_mechanism: {mechanism!r}")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
