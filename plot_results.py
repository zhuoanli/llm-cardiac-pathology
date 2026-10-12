"""
Generate the results figure: confusion matrix (Calibrated CoT) + accuracy bar chart.
Run after llm_cardiac_classify.py has produced results/llm_results.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

CLASSES = ["NOR", "DCM", "HCM", "MINF", "RV"]
RESULTS_FILE = Path("results/llm_results.json")
OUT_FIG = Path("figures/results.png")

with open(RESULTS_FILE) as f:
    data = json.load(f)

records = data["records"]
conditions_meta = data["conditions"]

# ---- Figure layout ----
fig = plt.figure(figsize=(13, 5.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.38)

# ---- Left: Confusion matrix for Calibrated CoT (E_GT_CalibCoT) ----
cond_key = "E_GT_CalibCoT"
cond_records = [r for r in records if r["condition"] == cond_key]

cm = np.zeros((5, 5), dtype=int)
for r in cond_records:
    i = CLASSES.index(r["gt"])
    j = CLASSES.index(r["pred"])
    cm[i, j] += 1

ax_cm = fig.add_subplot(gs[0])
im = ax_cm.imshow(cm, cmap="Blues", vmin=0, vmax=4)
ax_cm.set_xticks(range(5))
ax_cm.set_yticks(range(5))
ax_cm.set_xticklabels(CLASSES, fontsize=11)
ax_cm.set_yticklabels(CLASSES, fontsize=11)
ax_cm.set_xlabel("Predicted Label", fontsize=12)
ax_cm.set_ylabel("Ground Truth", fontsize=12)
ax_cm.set_title("Confusion Matrix\n(Calibrated CoT, GT metrics)", fontsize=11.5)
for i in range(5):
    for j in range(5):
        val = cm[i, j]
        color = "white" if val >= 3 else "black"
        ax_cm.text(j, i, str(val), ha="center", va="center",
                   fontsize=14, fontweight="bold", color=color)
plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

# ---- Right: Accuracy bar chart (5 key methods for paper) ----
ax_bar = fig.add_subplot(gs[1])

# Paper-relevant methods only (GT conditions + baselines)
methods = [
    "Random",
    "EF-Rule",
    "Direct\n(GT)",
    "Naive CoT\n(GT)",
    "Calibrated\nCoT (GT)",
]
accs = [
    data["baselines"]["random"]["accuracy"],
    data["baselines"]["ef_threshold"]["accuracy"],
    conditions_meta["A_GT_Direct"]["accuracy"],
    conditions_meta["B_GT_CoT"]["accuracy"],
    conditions_meta["E_GT_CalibCoT"]["accuracy"],
]
macro_f1s = [
    None,
    data["baselines"]["ef_threshold"].get("macro_f1"),
    conditions_meta["A_GT_Direct"]["macro_f1"],
    conditions_meta["B_GT_CoT"]["macro_f1"],
    conditions_meta["E_GT_CalibCoT"]["macro_f1"],
]

colors = [
    "#c8c8c8",   # Random: light gray
    "#a0a0a0",   # EF-rule: gray
    "#90c4e8",   # Direct: light blue
    "#f5a87a",   # Naive CoT: light orange (worse than direct)
    "#2196F3",   # Calibrated CoT: strong blue (best)
]
edge_colors = ["#888", "#666", "#1565C0", "#d84315", "#0d47a1"]

x = np.arange(len(methods))
bars = ax_bar.bar(x, [a * 100 for a in accs], color=colors,
                  edgecolor=edge_colors, linewidth=1.2, width=0.62)

# Annotate bars with accuracy %
for bar, acc, f1 in zip(bars, accs, macro_f1s):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{acc:.0%}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold")
    if f1 is not None:
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"F1={f1:.2f}", ha="center", va="center",
                    fontsize=8.5, color="white" if acc > 0.45 else "black",
                    fontweight="bold")

ax_bar.axhline(20, color="#999", linestyle="--", linewidth=1.0, label="Chance (20%)")
ax_bar.set_ylim(0, 80)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(methods, fontsize=10)
ax_bar.set_ylabel("5-class Accuracy (%)", fontsize=12)
ax_bar.set_title("Zero-Shot Cardiac Pathology Classification\n(ACDC, n=20, GT metrics)", fontsize=11.5)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.legend(fontsize=9, loc="upper left")

# Highlight "Calibrated CoT" bar with star annotation
best_bar = bars[4]
ax_bar.annotate("★ Best", xy=(best_bar.get_x() + best_bar.get_width() / 2,
                               best_bar.get_height() + 4.5),
                ha="center", fontsize=10, color="#0d47a1", fontweight="bold")

plt.suptitle("LLM-Based Zero-Shot Cardiac Pathology Classification from Segmentation Metrics",
             fontsize=12, y=1.02)
OUT_FIG.parent.mkdir(exist_ok=True)
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"Figure saved to {OUT_FIG}")
