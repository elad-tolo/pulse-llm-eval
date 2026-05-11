"""Standalone legend PDF for the test-with-train-side plots.

Produces a small figure containing just the legend (UCB-E green,
PULSE blue, Naive-Pooling red), saved to both plots_bbh/ and
plots_MMLU/."""
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ENTRIES = [
    ("PULSE",         "#4C72B0"),
    ("UCB-E",         "#55A868"),
    ("Naive-Pooling", "#C44E52"),
]

handles = [
    Line2D([0], [0], color=c, linewidth=2.0, label=lbl)
    for lbl, c in ENTRIES
]

fig = plt.figure(figsize=(2.5, 0.3))
fig.legend(
    handles=handles, loc="center", frameon=True, framealpha=0.95,
    fontsize=5, handlelength=1.0, ncol=len(ENTRIES),
)
plt.axis("off")

for out_dir in ("plots_bbh", "plots_MMLU"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/legend.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")

plt.close(fig)
