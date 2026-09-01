"""Generate clean method-comparison architecture figures for the IEEE paper.

Clean stacked-vertical design with no overlapping text.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)


def add_box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#000000", fontsize=9, weight="normal", lw=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          linewidth=lw, facecolor=fc, edgecolor=ec)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight)


def add_arrow(ax, x1, y1, x2, y2, color="#444444"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                             mutation_scale=12, color=color, linewidth=1.0)
    ax.add_patch(arrow)


def make_method_panel(ax, title, components, base_color, adapter_color, quant_color, base_fc):
    """components: list of dicts with keys: text, x, y, w, h, fc, ec, fontsize, weight, side_label"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, weight="bold", pad=8)

    # 1) Input (bottom-left)
    add_box(ax, 0.3, 1.0, 1.4, 0.7, "Input x", fc="#F1F3F4", ec="#5F6368", fontsize=9)

    # 2) Parallel paths: adapter (top) and base (bottom)
    add_box(ax, 0.3, 4.5, 2.0, 1.0, components["adapter_main"], fc=adapter_color, ec="#202124", weight="bold", fontsize=9)
    add_box(ax, 0.3, 3.2, 2.0, 0.7, components["adapter_label"], fc=adapter_color, ec="#5F6368", fontsize=8)

    add_box(ax, 0.3, 1.5, 2.0, 1.0, components["base_main"], fc=base_color, ec="#202124", weight="bold", fontsize=9)
    add_box(ax, 0.3, 0.2, 2.0, 0.7, components["base_label"], fc=quant_color, ec="#5F6368", fontsize=8)

    # 3) Sum
    add_box(ax, 3.0, 2.5, 0.8, 0.8, "+", fc="#FFFFFF", ec="#5F6368", fontsize=14, weight="bold", lw=1.2)

    # 4) Output
    add_box(ax, 4.5, 2.5, 1.6, 0.8, "Output y", fc="#F1F3F4", ec="#5F6368", fontsize=9)

    # 5) Side panel (right): explanations
    add_box(ax, 7.0, 0.5, 2.8, 8.5, components["side_text"], fc=base_fc, ec="#CCCCCC", fontsize=8)

    # Arrows
    add_arrow(ax, 1.7, 5.0, 3.0, 3.2)
    add_arrow(ax, 1.7, 2.0, 3.0, 2.8)
    add_arrow(ax, 3.8, 2.9, 4.5, 2.9)
    add_arrow(ax, 0.4, 1.35, 1.0, 2.0)
    add_arrow(ax, 0.4, 1.35, 1.0, 4.5)


fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

make_method_panel(axes[0],
    title="(a) LoRA",
    components={
        "adapter_main": "LoRA Adapter",
        "adapter_label": "A (FP32), B (FP32)\nrank r=8",
        "base_main": "FP32 Conv2d",
        "base_label": "Frozen pretrained weights\nNo quantization",
        "side_text": ("Standard LoRA\n\n"
                      "• Adapter factors A, B\n  in FP32\n\n"
                      "• Base conv: FP32,\n  frozen\n\n"
                      "• No quantization of\n  any kind\n\n"
                      "• Output: y = base(x)\n  + (B A)(x)\n\n"
                      "• At init: B = 0, so\n  y = base(x) exactly"),
    },
    base_color="#FCE8E6", adapter_color="#FCE8E6", quant_color="#FFFFFF", base_fc="#FFF5F4")

make_method_panel(axes[1],
    title="(b) QLoRA",
    components={
        "adapter_main": "LoRA Adapter",
        "adapter_label": "A (FP32), B (FP32)\nrank r=8",
        "base_main": "NF4 Conv2d",
        "base_label": "Frozen in 4-bit NF4\nBF16 dequant at compute",
        "side_text": ("QLoRA (Dettmers 2023)\n\n"
                      "• Adapter: FP32 LoRA\n  (same as LoRA)\n\n"
                      "• Base conv: 4-bit NF4\n  via bitsandbytes\n\n"
                      "• Dequant to BF16\n  each forward pass\n\n"
                      "• Compute dtype: BF16\n  (paper-mandated)\n\n"
                      "• Storage: ~7.6 MB"),
    },
    base_color="#FEF7E0", adapter_color="#FCE8E6", quant_color="#FEF7E0", base_fc="#FFFBF0")

make_method_panel(axes[2],
    title="(c) QA-LoRA (Ours)",
    components={
        "adapter_main": "Grouped LoRA",
        "adapter_label": "A (L x r), B (C_out x r)\nL=4 groups, rank r=8",
        "base_main": "INT4 Conv2d",
        "base_label": "True INT4 [-8, 7]\nLearnable scale & zp per group",
        "side_text": ("QA-LoRA (Xu 2024)\nadapted to 2D convs\n\n"
                      "• Adapter: grouped A\n  (L x r), standard B\n\n"
                      "• Base conv: group-wise\n  INT4 quantization\n\n"
                      "• Per-group learnable\n  scale alpha_j,\n  zero-point beta_j\n\n"
                      "• True-quant base,\n  dequant with learned\n  scale/zp in forward\n\n"
                      "• Optional zero-point\n  merge: deployable as\n  pure INT4 (validated\n  in our released code)"),
    },
    base_color="#E6F4EA", adapter_color="#E6F4EA", quant_color="#E8F0FE", base_fc="#F1F8F4")

plt.tight_layout()
out = os.path.join(OUT_DIR, "method_diagrams.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
