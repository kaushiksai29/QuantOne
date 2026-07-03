"""Charts from summary/summary.json ALONE (no scores/results access):
  1. charts/headline.png    — % reliability retained vs quant level, one line
                              per model, asymmetric error bars from delta CIs.
  2. charts/per_metric.png  — small multiples: absolute means per metric,
                              CI error bars on non-baseline quants.

  python report/make_charts.py [--summary summary/summary.json] [--out charts]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUANT_ORDER = ["FP16", "Q8_0", "Q4_K_M", "Q3_K_M"]
HEADLINE_METRIC = "schema_compliance"


def load_summary(path):
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    cells = {(c["model"], c["quant"], c["metric"]): c for c in s["cells"]}
    deltas = {(d["model"], d["quant"], d["metric"]): d for d in s["deltas"]}
    return s, cells, deltas


def _quants_present(cells, model, metric):
    return [q for q in QUANT_ORDER if (model, q, metric) in cells]


def headline_chart(summary, cells, deltas, out_path):
    baseline = summary["meta"]["baseline"]
    models = sorted({m for m, _, _ in cells})
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for model in models:
        if (model, baseline, HEADLINE_METRIC) not in cells:
            print(f"  headline: skipping {model} (no {baseline} baseline)")
            continue
        b = cells[(model, baseline, HEADLINE_METRIC)]["mean"]
        if b == 0:
            print(f"  headline: skipping {model} (baseline mean is 0)")
            continue
        xs, ys, lo_err, hi_err = [], [], [], []
        for q in _quants_present(cells, model, HEADLINE_METRIC):
            mean = cells[(model, q, HEADLINE_METRIC)]["mean"]
            xs.append(q)
            ys.append(100 * mean / b)
            d = deltas.get((model, q, HEADLINE_METRIC))
            if d:  # CI on the delta -> CI on retained%
                lo_err.append(100 * (d["delta"] - d["ci_lo"]) / b)
                hi_err.append(100 * (d["ci_hi"] - d["delta"]) / b)
            else:  # the baseline point itself
                lo_err.append(0.0)
                hi_err.append(0.0)
        ax.errorbar(xs, ys, yerr=[lo_err, hi_err], marker="o", capsize=4,
                    label=model)
        plotted = True
    if not plotted:
        print("  headline: nothing to plot")
        return
    ax.axhline(100, color="gray", lw=0.8, ls="--")
    ax.set_ylabel(f"% {HEADLINE_METRIC} retained vs {baseline}")
    ax.set_xlabel("quantization level")
    ax.set_title("Structured-output reliability retained under quantization\n"
                 "(paired bootstrap 95% CIs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def per_metric_chart(summary, cells, deltas, out_path):
    metrics = sorted({met for _, _, met in cells})
    models = sorted({m for m, _, _ in cells})
    ncols = min(3, len(metrics))
    nrows = -(-len(metrics) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)
    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        for model in models:
            xs = _quants_present(cells, model, metric)
            if not xs:
                continue
            ys = [cells[(model, q, metric)]["mean"] for q in xs]
            lo_err, hi_err = [], []
            for q in xs:
                d = deltas.get((model, q, metric))
                lo_err.append(d["delta"] - d["ci_lo"] if d else 0.0)
                hi_err.append(d["ci_hi"] - d["delta"] if d else 0.0)
            ax.errorbar(xs, ys, yerr=[lo_err, hi_err], marker="o",
                        capsize=3, label=model)
        ax.set_title(metric)
        ax.set_ylim(-0.02, 1.02)
    for j in range(len(metrics), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(5, max(1, len(models))))
    fig.suptitle("Per-metric means by quantization (95% CI error bars)")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="summary/summary.json")
    ap.add_argument("--out", default="charts")
    args = ap.parse_args()

    summary, cells, deltas = load_summary(args.summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    headline_chart(summary, cells, deltas, out / "headline.png")
    per_metric_chart(summary, cells, deltas, out / "per_metric.png")


if __name__ == "__main__":
    main()
