#!/usr/bin/env python3
"""Regenerate images/modelX (network-architecture diagram) for the paper.

The previous figure incorrectly showed the final layer's output shape as (1);
the classifier has 10 output classes. Layer names follow the PyTorch model in
AlexDominikCOPA2025.ipynb (SimpleMLP).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

LAYERS = [
    ("Flatten", "", "(28, 28)", "(784)"),
    ("Linear", "ReLU", "(784)", "(128)"),
    ("Dropout", "p = 0.2", "(128)", "(128)"),
    ("Linear", "", "(128)", "(10)"),
]

BOX_W, BOX_H, HEAD_H, GAP = 6.2, 1.15, 0.55, 0.9

fig, ax = plt.subplots(figsize=(18, 1.8))
ax.set_xlim(0, len(LAYERS) * (BOX_W + GAP) - GAP)
ax.set_ylim(0, BOX_H + HEAD_H)
ax.axis("off")

for i, (name, note, in_shape, out_shape) in enumerate(LAYERS):
    x0 = i * (BOX_W + GAP)
    # header bar
    ax.add_patch(Rectangle((x0, BOX_H), BOX_W, HEAD_H, facecolor="black"))
    title = name if not note else f"{name}  ({note})"
    ax.text(x0 + BOX_W / 2, BOX_H + HEAD_H / 2, title, color="white",
            ha="center", va="center", fontsize=15, fontweight="bold")
    # body: input / output cells
    for j, (label, val) in enumerate((("Input shape:", in_shape),
                                      ("Output shape:", out_shape))):
        cx = x0 + j * BOX_W / 2
        ax.add_patch(Rectangle((cx, 0), BOX_W / 2, BOX_H, facecolor="white",
                               edgecolor="black", linewidth=1.5))
        ax.text(cx + 0.18, BOX_H / 2, label, ha="left", va="center", fontsize=12)
        ax.text(cx + BOX_W / 2 - 0.18, BOX_H / 2, val, ha="right",
                va="center", fontsize=12, fontweight="bold")
    # arrow to next box
    if i < len(LAYERS) - 1:
        ax.add_patch(FancyArrow(x0 + BOX_W + 0.08, (BOX_H + HEAD_H) / 2,
                                GAP - 0.35, 0, width=0.03, head_width=0.22,
                                head_length=0.25, color="black"))

fig.tight_layout(pad=0.2)
for ext in ("png", "eps"):
    fig.savefig(f"../AMIS_submission/images/modelX.{ext}", dpi=200,
                bbox_inches="tight")
print("saved modelX.png / modelX.eps")
