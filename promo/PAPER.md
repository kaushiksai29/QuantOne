# Quantization and the Reliability of Structured Output in Small Language Models: A Controlled 30,000-Generation Study

**Author:** Kaushik Sai
**Artifacts:** Code — github.com/kaushiksai29/QuantOne · Dataset — hf.co/datasets/kash-on-the-dash/quantone

## Abstract

Post-training weight quantization is the default deployment strategy for open-weight language models, yet its effect on *structured-output reliability* — the ability to emit schema-valid JSON and correct tool calls — has been characterized largely by folklore rather than measurement. We evaluate five instruction-tuned models (1.7B–3.8B parameters) across four GGUF quantization levels (FP16, Q8_0, Q4_K_M, Q3_K_M) on 500 programmatically generated, machine-checkable tasks with three replicates each (30,000 generations). All scoring is deterministic; no model-based judging is used. Quantization-versus-FP16 differences are assessed with paired bootstrap 95% confidence intervals over task items. We find (1) Q8_0 produces no statistically significant degradation on any model or metric (0/25 comparisons); (2) Q4_K_M effects are small and inconsistent in direction; (3) Q3_K_M significantly degrades schema compliance in three of five models (largest: −18.6 points, Qwen2.5-3B); and (4) at Q3_K_M, two model families largely lose the ability to *decline* tool calls when no offered tool applies (Gemma-2-2B: 83.3% → 40.0%; Phi-3.5-mini: 39.2% → 0.0%). We additionally document two methodological hazards: a quant-correlated output-verbosity artifact that manufactures spurious "quantization helps" effects under strict parsing, and pervasive nondeterminism of greedy decoding on commodity inference stacks (12–27% of cells diverge across identical configurations). The entire study runs on free-tier hardware; all raw outputs, hashes, and seeds are published.

## 1. Introduction

Weight-only quantization reduces language-model memory footprints by 2–5× and is applied nearly universally when serving open-weight models via llama.cpp and the GGUF format (Gerganov et al., 2023/2025). Community guidance commonly asserts that 8-bit quantization is lossless in practice and 4-bit "close to free," based principally on perplexity deltas and general-benchmark accuracy (e.g., Dettmers et al., 2022; Frantar et al., 2022; Lin et al., 2024). However, production LLM systems increasingly depend on a narrower capability: *structured transduction* — producing output that a machine, not a human, consumes. A JSON object that is 95% correct is 0% usable. Whether quantization damage concentrates in this regime is an empirical question that perplexity does not answer, since a small uniform increase in token-level error can have super-linear effects on the probability that an entire structured emission is valid.

We ask a deliberately narrow question: **holding model, prompt, and decoding fixed, does GGUF quantization level measurably change structured-output reliability in small instruction models?**

## 2. Related Work

Post-training quantization methods for LLMs include 8-bit matrix decomposition (Dettmers et al., 2022), second-order weight rounding (Frantar et al., 2022), and activation-aware scaling (Lin et al., 2024); the k-quant schemes evaluated here are the llama.cpp community's widely deployed implementations (Gerganov et al., 2023/2025). Tool-use evaluation is anchored by the Berkeley Function-Calling Leaderboard (Yan et al., 2024), which our task families (selection, argument fidelity, and *irrelevance detection*) parallel at smaller scale with fully synthetic, contamination-free items. Our statistical protocol follows standard bootstrap methodology (Efron & Tibshirani, 1994). The models studied are documented in their respective reports (Abdin et al., 2024; Gemma Team et al., 2024; Grattafiori et al., 2024; Qwen Team, 2025; Allal et al., 2025).

## 3. Method

**Models and quantizations.** Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, Gemma-2-2B-it, Phi-3.5-mini-instruct, and SmolLM2-1.7B-Instruct, each in FP16, Q8_0, Q4_K_M, and Q3_K_M, each model's four files drawn from a single quantizer repository with SHA-256 hashes pinned before any run. Chat templates are the GGUF-embedded templates. Inference: llama-cpp-python 0.3.19, temperature 0, max 512 tokens, on Kaggle/Colab T4 GPUs.

**Tasks (n = 500).** Programmatically generated from a fixed seed: 350 schema-constrained generation tasks (JSON Schema draft 2020-12; nesting depth 1–4; enums, ISO-date patterns, bounded numerics, arrays, nested objects; `additionalProperties: false`), 120 tool-selection tasks (one correct tool among 2–5 distractors; argument values stated verbatim in the request), and 30 *should-not-call* tasks in which no offered tool applies. Gold labels exist by construction; generation is byte-reproducible.

**Metrics.** JSON parse rate; schema compliance; tool-selection accuracy; exact argument match; correct-decline rate. All validators are deterministic programs; no LLM judging is used anywhere (validators and unit tests are published).

**Parsing rule.** An output is parsed as the first complete JSON value in the emission (via incremental decoding that respects string literals and nesting); trailing text is ignored. Section 5.1 motivates this choice.

**Statistics.** For each (model, quantization, metric): seed replicates are averaged per item, then the quantization-vs-FP16 delta is assessed by paired bootstrap over the 500 items (10,000 resamples, fixed RNG). A difference is reported significant only when the 95% CI excludes zero. With 15 significant results among 75 comparisons under this criterion, false-positive contamination cannot be excluded for isolated small effects, but the headline results below are far from threshold and consistent in direction.

## 4. Results

**Q8_0 is statistically indistinguishable from FP16** on every model and metric (0/25 significant comparisons).

**Q4_K_M is nearly free.** Five significant comparisons of 25, small and mixed in sign (e.g., Qwen tool selection −3.1 points; Gemma schema compliance −4.4; Phi correct-decline −12.2; two *positive* deltas for Llama and SmolLM argument matching).

**Q3_K_M degrades schema compliance** in Qwen2.5-3B (83.5% → 65.0%; Δ = −18.6, CI [−23.9, −13.2]), Gemma-2-2B (77.3% → 65.9%; Δ = −11.4, CI [−16.5, −6.4]), and SmolLM2 (Δ = −5.0, CI [−9.5, −0.5]).

**Q3_K_M collapses refusal behavior in two families.** Correct-decline rate falls from 83.3% to 40.0% in Gemma-2-2B (Δ = −43.3, CI [−60.0, −26.7]) and from 39.2% to 0.0% in Phi-3.5-mini (Δ = −38.9, CI [−55.6, −22.2]). The failure direction is uniform: quantized models *substitute a plausible-but-wrong tool call for a refusal*. Because agent frameworks typically execute whatever call is emitted, this failure is silent at the system level.

**Model identity dominates quantization level.** Llama-3.2-3B exhibits weak deep-nesting compliance (27–35%) at *every* precision, frequently terminating mid-object; SmolLM2 never declines at any precision (0% across the board). The between-model spread on every metric exceeds any within-model quantization effect except the Q3 decline collapse.

## 5. Discussion

### 5.1 A quant-correlated measurement artifact

Phi-3.5-mini frequently emits a correct JSON answer and then continues generating (fabricating further dialogue turns until the token limit). Under a strict "the output must be exactly one JSON document" parser, these emissions score as failures — and the continuation rate is *quant-correlated* (53% of outputs truncated at FP16 vs. 8% at Q3_K_M), which manufactured a spurious "Q3 improves Phi tool selection by +27 points" result in our initial scoring. Re-scoring with first-JSON-value extraction eliminated the artifact while leaving non-verbose models bit-identical. We suggest that evaluation pipelines that report quantization (or sampling) effects on pass-rates without controlling for output-termination behavior are at risk of this class of confound. Because generation and scoring are separated stages with immutable raw outputs, the correction required no re-inference.

### 5.2 Greedy decoding is not deterministic in practice

Across the study, 12–27% of (quantization, item) cells produced non-identical outputs between replicates despite temperature 0 and identical configuration, attributable to non-associative floating-point reduction under threaded/GPU execution flipping argmax decisions at near-ties. Replicates in "deterministic" decoding regimes should be treated as draws from a noise distribution, as our seed-averaging protocol does.

### 5.3 Practical guidance

For structured-output workloads on small models: Q8_0 can be adopted without measurable penalty; Q4_K_M is a reasonable default; Q3_K_M should be validated per-model before deployment, with specific attention to refusal/irrelevance behavior, which degrades disproportionately and silently.

## 6. Limitations

Models ≤3.8B; a single inference stack (llama.cpp) and GPU class (T4); one task suite emphasizing schema compliance and single-turn tool calls; k-quant GGUF schemes only (not GPTQ/AWQ/imatrix variants); quantizer provenance varies across (never within) models; the lenient parsing rule measures usable output rather than strict format instruction-following. Effects at 7B+ scale may differ in either direction.

## 7. Reproducibility

All task-generation seeds, GGUF SHA-256 hashes, decoding parameters, bootstrap seed, and pipeline version are pinned in the repository. The dataset release contains all 30,000 raw outputs, per-item scores, and the aggregate summary. A fresh-environment script regenerates any (model, quantization) cell from scratch and verifies agreement with published means within replicate noise (±0.03).

## References

Abdin, M., Aneja, J., Awadalla, H., Awadallah, A., Awan, A. A., Bach, N., ... Zhou, X. (2024). *Phi-3 technical report: A highly capable language model locally on your phone* (arXiv:2404.14219). arXiv. https://arxiv.org/abs/2404.14219

Allal, L. B., Lozhkov, A., Bakouch, E., Blázquez, G. M., Penedo, G., Tunstall, L., ... Wolf, T. (2025). *SmolLM2: When Smol goes big — Data-centric training of a small language model* (arXiv:2502.02737). arXiv. https://arxiv.org/abs/2502.02737

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. *Advances in Neural Information Processing Systems, 35*. https://arxiv.org/abs/2208.07339

Efron, B., & Tibshirani, R. J. (1994). *An introduction to the bootstrap*. Chapman & Hall/CRC.

Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). *GPTQ: Accurate post-training quantization for generative pre-trained transformers* (arXiv:2210.17323). arXiv. https://arxiv.org/abs/2210.17323

Gemma Team, Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., ... Kenealy, K. (2024). *Gemma 2: Improving open language models at a practical size* (arXiv:2408.00118). arXiv. https://arxiv.org/abs/2408.00118

Gerganov, G., & llama.cpp contributors. (2023–2025). *llama.cpp* [Computer software]. GitHub. https://github.com/ggml-org/llama.cpp

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., ... Ma, Z. (2024). *The Llama 3 herd of models* (arXiv:2407.21783). arXiv. https://arxiv.org/abs/2407.21783

Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., ... Han, S. (2024). AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration. *Proceedings of Machine Learning and Systems, 6*. https://arxiv.org/abs/2306.00978

Qwen Team. (2025). *Qwen2.5 technical report* (arXiv:2412.15115). arXiv. https://arxiv.org/abs/2412.15115

Yan, F., Mao, H., Ji, C. C.-J., Zhang, T., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2024). *Berkeley Function Calling Leaderboard*. https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html
