"""Bootstrap must recover a planted effect on synthetic data (Phase 4
acceptance) and never claim significance on an exact null."""

import numpy as np
import pandas as pd

from scorer.aggregate import compute_summary, seed_averaged

N_PROMPTS = 300
SEEDS = 3


def _synthetic_scores(q4_degrade):
    """Paired synthetic data: FP16 succeeds w.p. 0.9; Q4_K_M keeps each FP16
    success only w.p. (1 - q4_degrade). True delta = -0.9 * q4_degrade."""
    rng = np.random.default_rng(7)
    rows = []
    for p in range(N_PROMPTS):
        pid = f"struct-{p:04d}"
        for seed in range(SEEDS):
            fp16 = float(rng.random() < 0.9)
            q4 = fp16 * float(rng.random() >= q4_degrade)
            for quant, value in (("FP16", fp16), ("Q4_K_M", q4)):
                rows.append({"model": "m", "quant": quant, "prompt_id": pid,
                             "family": "structured_output", "seed": seed,
                             "pipeline_version": 1,
                             "metric": "schema_compliance", "value": value})
    return pd.DataFrame(rows)


def test_bootstrap_recovers_planted_effect():
    df = _synthetic_scores(q4_degrade=0.15)
    summary = compute_summary(df, n_boot=10_000)
    (d,) = summary["deltas"]
    true_delta = -0.9 * 0.15
    assert d["significant"] is True
    assert d["ci_hi"] < 0                      # CI excludes zero
    assert d["ci_lo"] < true_delta < d["ci_hi"]  # CI covers the true effect
    assert abs(d["delta"] - true_delta) < 0.04
    assert d["n_prompts"] == N_PROMPTS


def test_exact_null_is_never_significant():
    df = _synthetic_scores(q4_degrade=0.0)  # Q4 identical to FP16 per seed
    summary = compute_summary(df, n_boot=2_000)
    (d,) = summary["deltas"]
    assert d["delta"] == 0.0
    assert d["significant"] is False


def test_seeds_are_averaged_before_bootstrap():
    df = pd.DataFrame([
        {"model": "m", "quant": "FP16", "prompt_id": "p0", "seed": s,
         "metric": "schema_compliance", "value": float(s % 2),
         "family": "structured_output", "pipeline_version": 1}
        for s in range(4)])
    per_prompt = seed_averaged(df)
    assert len(per_prompt) == 1
    assert per_prompt["value"].iloc[0] == 0.5


def test_missing_baseline_yields_no_deltas():
    df = _synthetic_scores(0.15)
    df = df[df["quant"] != "FP16"]
    summary = compute_summary(df)
    assert summary["deltas"] == []
    assert len(summary["cells"]) == 1  # means still reported


def test_summary_is_reproducible():
    df = _synthetic_scores(0.15)
    s1 = compute_summary(df, n_boot=1_000)
    s2 = compute_summary(df, n_boot=1_000)
    assert s1["deltas"] == s2["deltas"]
    assert s1["cells"] == s2["cells"]
