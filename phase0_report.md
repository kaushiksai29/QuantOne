# QuantOne v2 — Phase 0 report (⛔ checkpoint)

Status: manifest verified against live HF API (2026-07-10), `manifest.json`
generated with per-file SHA-256 + revisions. GPU probes (template check,
thinking-off verification, tok/s) run via `notebooks/v2_phase0_probe.ipynb`
on Kaggle (small tier) and Colab (big tier) — NOT locally, per operator
preference. Results land in `phase0_gpu_report_{tier}.json` and must be
reviewed before Phase 1.

## Model verification vs the plan

| plan id | plan name | verified reality |
|---|---|---|
| qwen35-2b | Qwen3.5-2B-Instruct | ✅ exists as **Qwen/Qwen3.5-2B** (no "-Instruct" suffix; "-Base" is the base variant) |
| qwen35-4b | Qwen3.5-4B-Instruct | ✅ **Qwen/Qwen3.5-4B** (same naming) |
| gemma4-e4b | Gemma 4 E4B-it | ✅ google/gemma-4-E4B-it + official QAT q4_0 GGUF |
| ministral3-3b | Ministral-3-3B-Instruct-2512 | ✅ exists |
| phi4-mini | Phi-4-mini-instruct | ✅ exists |
| smollm3-3b | SmolLM3-3B | ✅ exists |
| llama32-3b | Llama-3.2-3B-Instruct | ✅ v1 anchor, unchanged |
| qwen36-27b | Qwen3.6-27B | ✅ **Qwen/Qwen3.6-27B** (no separate -Instruct repo visible) |
| gemma4-qat | Gemma 4 14B | ❌ **no 14B exists** (sizes: E2B/E4B/12B/26B-A4B/31B). Plan's named fallback **26B-A4B-it** exists WITH official QAT q4_0 GGUF → selected. Note gemma-4-12B-it also has an official QAT GGUF if a dense model is preferred (decision D4). |

Kill list: empty — every model resolved. Discovery: Qwen3.5, Gemma-4-E4B,
and Ministral-3 are **multimodal** (mmproj files in GGUF repos); we use the
LM GGUFs only — text-only protocol, note in Methods.

## Quant source selection (one repo per model)

**bartowski is the only allowed source that publishes IQ3_M** — official
repos, ggml-org, unsloth and lmstudio-community all lack it. Priority order
(official > ggml-org > unsloth > bartowski) therefore yields bartowski for
most models: it's the only single-repo solution covering the full v2 ladder.
None of the selected repos are abliterated/uncensored.

| model | repo | coverage |
|---|---|---|
| qwen35_2b / qwen35_4b | bartowski | all 6 ✅ |
| gemma4_e4b | bartowski (+google QAT) | all 6 + QAT ✅ |
| ministral3_3b | bartowski | all 6 ✅ |
| smollm3_3b | bartowski | all 6 ✅ |
| phi4_mini | **decision D1** | no single repo has FP16 + IQ quants |
| llama32_3b | unsloth (v1 files) | v1 ladder + IQ4_XS; no IQ3_M (**D2**) |
| qwen36_27b | bartowski | all 5 big-tier ✅ |
| gemma4_26b_a4b | bartowski (+google QAT) | all 5 + QAT ✅ |

## ⛔ Decisions — RESOLVED 2026-07-12 (operator approved "start rolling")

- **D1 → (b)**: phi4_mini from bartowski, **Q8_0 baseline** (v1 Q8≡FP16
  justification, mirrors big tier). Keeps the imatrix headline complete.
- **D2 → accepted**: llama32_3b stays on unsloth v1 files, no IQ3_M cell.
- **D3 → defer to probe**: gemma4_e4b FP16 decided by measured T4 offload.
- **D4 → confirmed**: gemma4_26b_a4b (MoE) is the big-tier Gemma.
- **D5 → pin 0.3.33**: operator mandated zero local model execution, so the
  Windows-wheel CPU incompatibility is moot; newest release wins for Linux.
- **New (community-research adjustments, 2026-07-12):** every v2 output leads
  with a sweet-spot table ("at X GB VRAM run Y at Z"); Methods/Limitations
  explicitly scope out NVFP4/INT4-AutoRound/W4A4 (T4-incompatible) and
  future-work.md gains an NVFP4 successor-study entry.

## Original decision text (for the record)

- **D1 — phi4_mini source.** unsloth has FP16/Q8/Q4_K_M/Q3_K_M but **zero IQ
  quants**; bartowski has all IQ quants but **no FP16/BF16**.
  (a) unsloth: keeps FP16 baseline, loses the imatrix comparison for this model.
  (b) bartowski with **Q8_0 as phi4's baseline** (justified by v1's Q8≡FP16,
  mirroring the big tier): keeps the imatrix headline complete across models.
  My recommendation: **(b)** — the imatrix question is v2 headline #1; losing
  a model from it hurts more than a baseline substitution we've already
  validated.
- **D2 — llama32_3b IQ3_M.** The v1 drift control requires byte-identical
  unsloth files, and unsloth has no IQ3_M. Recommend: run llama32 without
  IQ3_M (IQ4_XS is available), keep it as the v1 anchor. Alternative of
  adding bartowski just for IQ3_M violates the one-repo rule.
- **D3 — gemma4_e4b FP16 = 15.1 GB** — same T4 squeeze v1 avoided by model
  choice. Options: (a) partial offload (slow, measure in probe), (b) Q8_0
  baseline for this model too, (c) accept slow FP16 cells. Probe data will
  quantify (a).
- **D4 — big-tier Gemma.** Plan fallback = 26B-A4B (MoE, official QAT ✅,
  selected). gemma-4-12B-it (dense, official QAT ✅) is a cleaner
  quantization story if you prefer dense over MoE; 26B-A4B is the stronger
  model. Confirm 26B-A4B or switch.
- **D5 — llama-cpp-python pin.** Recommend **0.3.28**: confirmed to load the
  Qwen3.5 architecture, and Windows wheels ≥0.3.30 crash on your laptop's CPU
  (illegal instruction — raised build baseline), so 0.3.28 is the newest
  version that can ever run locally if needed. The GPU probe notebook
  currently installs 0.3.33 (Linux wheels unaffected); if you approve 0.3.28
  as the study pin, I'll align the notebook before Phase 2 — one version
  everywhere is the cleaner Methods statement. Qwen3.6/Gemma-4/Ministral-3
  load support is verified by the probe notebook either way.

## Budget reality (pre-probe estimates; probe fills in measured tok/s)

- Total GGUF downloads across the study ≈ **324 GB** (streamed one quant at
  a time; peak disk = largest single file, 32.3 GB for qwen36 Q8_0 — fits
  Colab's disk, not Kaggle's ~20 GB /working: big tier MUST run on Colab, as
  planned).
- Small-tier core = 151,200 generations. At v1's observed 2–4 s/gen on T4 →
  ~85–170 T4-hours ≈ 3–6 weeks of Kaggle quota. The plan's ⛔ knobs (fewer
  seeds on schema tasks, IQ quants on the 500-subset only, drop Q8 for some
  models) will likely be needed — decide after probe tok/s lands.

## Next actions

1. You: push the `v2` branch (GitHub Desktop), then run
   `notebooks/v2_phase0_probe.ipynb` twice — Kaggle T4 with `TIER="small"`,
   Colab L4 with `TIER="big"` — and hand me both `phase0_gpu_report_*.json`.
2. Me: fold probe results into this report, finalize kill list + runtime
   projection, present the Phase 4 knob decision.
3. ⛔ You approve → Phase 1 (dataset v2: 1,200 tasks with the 200-item
   decline subset).
