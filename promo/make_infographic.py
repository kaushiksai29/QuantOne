"""Shareable summary-card infographic (PNG, 1200x1500) from summary.json."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, INK, MUTE = "#0f1420", "#f2f4f8", "#8b95a8"
GOOD, WARN, BAD, ACCENT = "#3fb96f", "#e0a93e", "#e05252", "#5b8def"

fig = plt.figure(figsize=(8, 10), dpi=150)
fig.patch.set_facecolor(BG)


def text(x, y, s, size, color=INK, weight="normal", ha="left", family="DejaVu Sans"):
    fig.text(x, y, s, size=size, color=color, weight=weight, ha=ha,
             family=family, va="top")


# ---- header ----
text(0.5, 0.975, "DOES QUANTIZATION BREAK", 26, ha="center", weight="bold")
text(0.5, 0.943, "STRUCTURED OUTPUT?", 26, ha="center", weight="bold", color=ACCENT)
text(0.5, 0.905, "Shrinking an LLM saves memory — like compressing a photo.", 12,
     color=MUTE, ha="center")
text(0.5, 0.885, "Does the model still produce valid JSON and correct tool calls?", 12,
     color=MUTE, ha="center")
text(0.5, 0.856,
     "30,000 generations  ·  5 models  ·  4 compression levels  ·  $0 compute",
     12.5, ha="center", weight="bold")

# ---- verdict rows ----
verdicts = [
    ("Q8  (half the size)", "FREE", "0 of 25 comparisons got worse", GOOD),
    ("Q4  (quarter size)", "NEARLY FREE", "a few small effects, nothing consistent", WARN),
    ("Q3  (fifth the size)", "BREAKS", "JSON rule-following drops in 3 of 5 models", BAD),
]
y = 0.79
for name, verdict, note, color in verdicts:
    ax = fig.add_axes([0.06, y - 0.052, 0.88, 0.062])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                fc="#1a2233", ec=color, lw=2,
                                transform=ax.transAxes))
    text(0.10, y - 0.008, name, 15, weight="bold")
    text(0.60, y - 0.008, verdict, 15, color=color, weight="bold")
    text(0.10, y - 0.033, note, 11, color=MUTE)
    y -= 0.085

# ---- the headline chart: decline collapse ----
text(0.5, 0.525, "The scary part: compressed models stop saying “no”", 15,
     ha="center", weight="bold")
text(0.5, 0.500, "% of the time the model correctly refuses when NO available tool fits the request",
     10.5, color=MUTE, ha="center")

ax = fig.add_axes([0.12, 0.24, 0.76, 0.24])
ax.set_facecolor(BG)
models = ["Gemma-2-2B", "Phi-3.5-mini"]
full = [83.3, 39.2]   # FP16 correct-decline %
q3 = [40.0, 0.0]      # Q3_K_M
x = [0, 1]
w = 0.32
b1 = ax.bar([i - w / 2 - 0.02 for i in x], full, w, color=ACCENT, label="Full precision")
b2 = ax.bar([i + w / 2 + 0.02 for i in x], q3, w, color=BAD, label="Q3 compressed")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{b.get_height():.0f}%", ha="center", size=12, color=INK,
                weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, size=12, color=INK)
ax.set_ylim(0, 100)
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(frameon=False, labelcolor=INK, fontsize=11, loc="upper right")
text(0.5, 0.215, "A model that can’t refuse will call the wrong tool instead — silently.", 12,
     ha="center", color=BAD, weight="bold")

# ---- method + footer ----
text(0.5, 0.165, "HOW IT WAS MEASURED", 11, ha="center", color=MUTE, weight="bold")
text(0.5, 0.143,
     "Every answer graded by a program (no AI judges) · differences claimed only when a", 10.5,
     color=MUTE, ha="center")
text(0.5, 0.125,
     "paired bootstrap 95% confidence interval excludes zero · all raw outputs published", 10.5,
     color=MUTE, ha="center")
text(0.5, 0.085, "github.com/kaushiksai29/QuantOne", 13, ha="center",
     color=ACCENT, weight="bold")
text(0.5, 0.062, "dataset + interactive results on Hugging Face: kash-on-the-dash/quantone",
     10.5, ha="center", color=MUTE)

fig.savefig("promo/infographic.png", facecolor=BG, bbox_inches="tight")
print("wrote promo/infographic.png")
