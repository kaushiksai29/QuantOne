"""QuantOne v2 Phase 0: builds manifest.json — for every (model, quant) cell,
the exact GGUF file, its SHA-256 (HF LFS oid), repo revision, and per-model
protocol fields (thinking-off mechanism, tier, arms). Verifies exactly one
file matches each required quant; anything unresolvable lands in the report's
kill/decision list rather than silently passing.

Run: python scripts/v2_build_manifest.py   -> manifest.json + stdout report
"""

import json
import re
import sys
import urllib.request
from datetime import date

SMALL_QUANTS = ["FP16", "Q8_0", "Q4_K_M", "IQ4_XS", "Q3_K_M", "IQ3_M"]
BIG_QUANTS = ["Q8_0", "Q4_K_M", "IQ4_XS", "Q3_K_M", "IQ3_M"]

# model_key -> (base_model, gguf_repo, quants, extras)
PLAN = {
    "qwen35_2b": {
        "base": "Qwen/Qwen3.5-2B", "repo": "bartowski/Qwen_Qwen3.5-2B-GGUF",
        "tier": "small", "quants": SMALL_QUANTS,
        "thinking_off": "hybrid thinking model — mechanism verified in local probe",
    },
    "qwen35_4b": {
        "base": "Qwen/Qwen3.5-4B", "repo": "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "tier": "small", "quants": SMALL_QUANTS,
        "thinking_off": "same mechanism as qwen35_2b",
        "arms": ["kv"],
    },
    "gemma4_e4b": {
        "base": "google/gemma-4-E4B-it",
        "repo": "bartowski/google_gemma-4-E4B-it-GGUF",
        "tier": "small", "quants": SMALL_QUANTS,
        "thinking_off": "n/a (no thinking mode) — verified in local probe",
        "arms": ["kv", "qat"],
        "qat_repo": "google/gemma-4-E4B-it-qat-q4_0-gguf",
    },
    "ministral3_3b": {
        "base": "mistralai/Ministral-3-3B-Instruct-2512",
        "repo": "bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF",
        "tier": "small", "quants": SMALL_QUANTS,
        "thinking_off": "n/a (no thinking mode) — verified in local probe",
    },
    "phi4_mini": {
        "base": "microsoft/Phi-4-mini-instruct",
        "repo": "unsloth/Phi-4-mini-instruct-GGUF",
        "tier": "small",
        # DECISION NEEDED: bartowski lacks FP16 for this model; unsloth lacks
        # BOTH IQ quants. Option A (below): unsloth, FP16 baseline, no IQ
        # cells. Option B: bartowski, all IQ quants, Q8_0 baseline (justified
        # by v1 Q8≡FP16, mirroring the big tier).
        "quants": ["FP16", "Q8_0", "Q4_K_M", "Q3_K_M"],
        "quant_gaps": {"IQ3_M": "unsloth has no IQ quants",
                       "IQ4_XS": "unsloth has no IQ quants"},
        "thinking_off": "n/a (no thinking mode)",
    },
    "smollm3_3b": {
        "base": "HuggingFaceTB/SmolLM3-3B",
        "repo": "bartowski/HuggingFaceTB_SmolLM3-3B-GGUF",
        "tier": "small", "quants": SMALL_QUANTS,
        "thinking_off": "/no_think system-prompt flag — verified in local probe",
    },
    "llama32_3b": {
        "base": "meta-llama/Llama-3.2-3B-Instruct",
        "repo": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "tier": "small",
        # DECISION: v1 anchor — unsloth files must stay byte-identical to v1
        # for the drift control; unsloth has no IQ3_M.
        "quants": ["FP16", "Q8_0", "Q4_K_M", "IQ4_XS", "Q3_K_M"],
        "quant_gaps": {"IQ3_M": "v1-anchor repo (unsloth) has no IQ3_M; "
                                "switching repos would break the drift control"},
        "thinking_off": "n/a",
        "v1_anchor": True,
    },
    "qwen36_27b": {
        "base": "Qwen/Qwen3.6-27B", "repo": "bartowski/Qwen_Qwen3.6-27B-GGUF",
        "tier": "big", "quants": BIG_QUANTS,
        "thinking_off": "hybrid thinking model — verify on first Colab cell; "
                        "known GGUF template issues, pin template hash",
    },
    "gemma4_26b_a4b": {
        "base": "google/gemma-4-26B-A4B-it",
        "repo": "bartowski/google_gemma-4-26B-A4B-it-GGUF",
        "tier": "big", "quants": BIG_QUANTS,
        "thinking_off": "n/a",
        "arms": ["qat"],
        "qat_repo": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
        "notes": "MoE (A4B): measure real memory in Phase 0 GPU probe",
    },
}

# quant token -> filename patterns to try, in order (bartowski/unsloth naming)
QUANT_PATTERNS = {
    "FP16": ["f16", "F16", "fp16", "BF16", "bf16"],
    "Q8_0": ["Q8_0"], "Q4_K_M": ["Q4_K_M"], "IQ4_XS": ["IQ4_XS"],
    "Q3_K_M": ["Q3_K_M"], "IQ3_M": ["IQ3_M"],
}


def repo_tree(repo):
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    return json.load(urllib.request.urlopen(url))


def repo_sha(repo):
    return json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/models/{repo}"))["sha"]


def find_quant_file(tree, quant):
    ggufs = [f for f in tree if f["path"].endswith(".gguf")
             and "imatrix" not in f["path"] and "mmproj" not in f["path"]]
    for pat in QUANT_PATTERNS[quant]:
        hits = [f for f in ggufs
                if re.search(rf"[-.]{re.escape(pat)}(-\d+-of-\d+)?\.gguf$",
                             f["path"])]
        if len(hits) == 1:
            return hits[0], pat
        if len(hits) > 1:  # split files (multi-part) — take part list
            return sorted(hits, key=lambda f: f["path"]), pat
    return None, None


def main():
    manifest = {"generated": str(date.today()), "pipeline_version": 2,
                "llama_cpp_python": "0.3.33", "models": {}}
    problems = []
    for key, spec in PLAN.items():
        tree = repo_tree(spec["repo"])
        rev = repo_sha(spec["repo"])
        entry = {"base_model": spec["base"], "gguf_repo": spec["repo"],
                 "revision": rev, "tier": spec["tier"],
                 "thinking_off": spec["thinking_off"],
                 "arms": spec.get("arms", []),
                 "quant_gaps": spec.get("quant_gaps", {}),
                 "notes": spec.get("notes", ""), "files": {}}
        for quant in spec["quants"]:
            f, pat = find_quant_file(tree, quant)
            if f is None:
                problems.append(f"{key}: no file for {quant} in {spec['repo']}")
                continue
            if isinstance(f, list):
                entry["files"][quant] = [
                    {"path": p["path"], "sha256": p["lfs"]["oid"],
                     "gb": round(p["size"] / 1e9, 2)} for p in f]
            else:
                entry["files"][quant] = {"path": f["path"],
                                         "sha256": f["lfs"]["oid"],
                                         "gb": round(f["size"] / 1e9, 2)}
        if "qat_repo" in spec:
            qtree = repo_tree(spec["qat_repo"])
            qrev = repo_sha(spec["qat_repo"])
            qf = [f for f in qtree if f["path"].endswith(".gguf")
                  and "mmproj" not in f["path"]]
            if len(qf) == 1:
                entry["qat"] = {"repo": spec["qat_repo"], "revision": qrev,
                                "path": qf[0]["path"],
                                "sha256": qf[0]["lfs"]["oid"],
                                "gb": round(qf[0]["size"] / 1e9, 2)}
            else:
                problems.append(f"{key}: QAT repo has {len(qf)} model ggufs")
        manifest["models"][key] = entry

    with open("manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    total_gb = 0
    for key, m in manifest["models"].items():
        sizes = []
        for q, v in m["files"].items():
            gb = sum(x["gb"] for x in v) if isinstance(v, list) else v["gb"]
            sizes.append(f"{q}={gb:.1f}GB")
            total_gb += gb
        qat = f" +QAT={m['qat']['gb']:.1f}GB" if "qat" in m else ""
        gaps = f"  GAPS: {list(m['quant_gaps'])}" if m["quant_gaps"] else ""
        print(f"{key:16s} [{m['tier']}] {' '.join(sizes)}{qat}{gaps}")
    print(f"\nTOTAL download across study: ~{total_gb:.0f} GB (one quant at a "
          f"time per session; peak disk = largest single file)")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(" -", p)
    print("\nwrote manifest.json")


if __name__ == "__main__":
    main()
