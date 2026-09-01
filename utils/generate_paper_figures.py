"""Generate clean method-comparison architecture figures for the IEEE paper."""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT_DIR = "paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)


def draw_box(ax, x, y, w, h, text, fc="#E8F0FE", ec="#1A73E8", fontsize=9, weight="normal"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          linewidth=1.0, facecolor=fc, edgecolor=ec)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color="#5F6368"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12,
                             color=color, linewidth=1.0)
    ax.add_patch(arrow)


def draw_panel(ax, title, base_label, adapter_box_text, adapter_label, quant_label, base_color, adapter_color, quant_color):
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, weight="bold", pad=10)

    # Input (bottom)
    draw_box(ax, 0.3, 3.0, 1.4, 0.9, "Input $x$", fc="#F1F3F4", ec="#5F6368")
    # Adapter box (middle)
    draw_box(ax, 2.4, 3.0, 1.7, 0.9, "$B A$", fc=adapter_color, ec="#202124", weight="bold", fontsize=10)
    # Quantized/frozen base (right)
    draw_box(ax, 5.0, 3.0, 2.5, 0.9, base_label, fc=base_color, ec="#202124", weight="bold", fontsize=10)
    # Add label (left)
    draw_box(ax, 4.0, 4.0, 1.4, 0.5, "+", fc="#FFFFFF", ec="#5F6368", fontsize=12)

    # Top label (adapter details)
    draw_box(ax, 2.4, 6.4, 1.7, 1.6, adapter_box_text, fc=adapter_color, ec="#5F6368", fontsize=8)
    # Top label (quant details)
    draw_box(ax, 5.0, 6.4, 2.5, 1.6, quant_label, fc=quant_color, ec="#5F6368", fontsize=8)

    # Output
    draw_box(ax, 8.0, 3.0, 1.5, 0.9, "Output $y$", fc="#F1F3F4", ec="#5F6368")

    # Arrows
    draw_arrow(ax, 1.7, 3.45, 2.4, 3.45)
    draw_arrow(ax, 4.1, 3.45, 5.0, 3.45)
    draw_arrow(ax, 7.5, 3.45, 8.0, 3.45)

    # Note at bottom
    draw_box(ax, 0.3, 0.0, 9.2, 1.6,
             "Pretrained backbone is FROZEN.\nOnly adapter parameters (and, for QA-LoRA,\nquantization scale/zp) receive gradients. At init, B=0 so output=base(x).",
             fc="#FFF8E1", ec="#F9AB00", fontsize=8)


fig, axes = plt.subplots(1, 3, figsize=(13, 5.0))

draw_panel(axes[0], "(a) LoRA",
           base_label="FP32 Conv2d\n(Frozen)",
           adapter_box_text="$A \in \mathbb{R}^{C_{in}\\times r}$ (FP32)\n$B \in \mathbb{R}^{r \\times C_{out}}$ (FP32)\nRank $r=8$",
           adapter_label="",
           quant_label="No quantization\nBase weights unchanged",
           base_color="#FCE8E6", adapter_color="#FCE8E6", quant_color="#FFFFFF")

draw_panel(axes[1], "(b) QLoRA",
           base_label="NF4 Conv2d\n(Frozen)",
           adapter_box_text="$A \in \mathbb{R}^{C_{in}\\times r}$ (FP32)\n$B \in \mathbb{R}^{r \\times C_{out}}$ (FP32)\nRank $r=8$",
           adapter_label="",
           quant_label="bitsandbytes NF4\nBF16 dequant at compute time",
           base_color="#FEF7E0", adapter_color="#FCE8E6", quant_color="#FEF7E0")

draw_panel(axes[2], "(c) QA-LoRA (Ours)",
           base_label="INT4 Conv2d\n(Frozen)",
           adapter_box_text="$A \in \mathbb{R}^{L\\times r}$ (Grouped)\n$B \in \mathbb{R}^{C_{out}\\times r}$\n$L=4$ groups, rank $r=8$",
           adapter_label="",
           quant_label="Group-wise INT4 [-8, 7]\nLearnable $\\alpha_j$, $\\beta_j$ per group\nTrue quantized base",
           base_color="#E6F4EA", adapter_color="#E6F4EA", quant_color="#E8F0FE")

plt.tight_layout()
out = os.path.join(OUT_DIR, "method_diagrams.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")


# Figure 2: Parameter efficiency vs accuracy scatter (cleaner)
from matplotlib.ticker import FuncFormatter, LogLocator


def k_format(x, pos):
    if x >= 1e6:
        return f"{x/1e6:g}M"
    if x >= 1e3:
        return f"{x/1e3:g}K"
    return f"{x:g}"


fig, ax = plt.subplots(figsize=(7.5, 4.5))

methods = [
    ("LoRA",         192816,  98.89, "#4285F4", (0.25, 0.25)),
    ("QLoRA",        192816,  98.91, "#FBBC04", (0.25, -0.30)),
    ("QA-LoRA",      241958,  99.52, "#34A853", (0.30, 0.05)),
    ("Full FT\n(cited)", 4249042, 99.56, "#EA4335", (-0.55, -0.05)),
]

for name, params, acc, color, (dx, dy) in methods:
    ax.scatter(params, acc, s=260, c=color, edgecolors="black", linewidth=1.2, zorder=3)
    ax.annotate(name, xy=(params, acc), xytext=(params * (1 + dx), acc + dy * 0.3),
                fontsize=10, weight="bold",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5))

ax.set_xscale("log")
ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[2.0, 3.0, 5.0], numticks=24))
ax.xaxis.set_major_formatter(FuncFormatter(k_format))
ax.xaxis.set_minor_formatter(FuncFormatter(k_format))
ax.tick_params(axis="x", which="minor", labelsize=7, labelcolor="gray")
ax.set_xlabel("Trainable parameters", fontsize=11)
ax.set_ylabel("Test accuracy on PlantVillage (%)", fontsize=11)
ax.set_title("Parameter efficiency: PEFT vs full fine-tuning on PlantVillage", fontsize=12, weight="bold")
ax.set_ylim(98.5, 99.8)
ax.set_xlim(1.5e5, 6e6)
ax.grid(True, alpha=0.3, linestyle="--", which="both")
ax.axhline(99.56, color="#EA4335", linestyle=":", alpha=0.5)
ax.axhline(99.52, color="#34A853", linestyle=":", alpha=0.5)
ax.text(1.1e6, 99.585, "Full FT cited = 99.56%", fontsize=9, color="#EA4335", ha="center", weight="bold")
ax.text(1.1e6, 99.51, "QA-LoRA = 99.52%", fontsize=9, color="#34A853", ha="center", weight="bold")

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "param_efficiency.png")
plt.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out2}")
