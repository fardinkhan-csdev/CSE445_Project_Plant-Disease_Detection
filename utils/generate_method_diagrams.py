"""Generate method diagrams arranged vertically: A, B, C stacked.

Labels use matplotlib text wrapping (soft wrap) so text never overflows its box.
"""
import os
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)


def add_box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#000000",
            fontsize=9, weight="normal", lw=1.0, wrap_chars=22):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         linewidth=lw, facecolor=fc, edgecolor=ec)
    ax.add_patch(box)
    # Hyphen-aware word wrap so tokens longer than wrap_chars are split.
    lines = []
    for line in text.split("\n"):
        if not line:
            lines.append("")
            continue
        # Insert zero-width break after every char if line has no spaces and is long
        if len(line) > wrap_chars and " " not in line:
            chunks = [line[i:i + wrap_chars] for i in range(0, len(line), wrap_chars)]
            lines.extend(chunks)
            continue
        wrapped = textwrap.wrap(line, width=wrap_chars, break_long_words=True,
                                break_on_hyphens=False) or [line]
        lines.extend(wrapped)
    final = "\n".join(lines)
    ax.text(x + w / 2, y + h / 2, final, ha="center", va="center",
            fontsize=fontsize, weight=weight)


def add_arrow(ax, x1, y1, x2, y2, color="#444444"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                            mutation_scale=12, color=color, linewidth=1.0)
    ax.add_patch(arrow)


def make_method_panel(ax, title, components, base_color, adapter_color, quant_color, base_fc):
    # Wider, shorter aspect so side panels don't clip on left/right.
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, weight="bold", pad=6)

    BOX_W = 3.0
    OUT_W = 2.0
    LABEL_H = 1.6

    # Input (left)
    add_box(ax, 0.3, 3.0, 1.4, 0.9, "Input x",
            fc="#F1F3F4", ec="#5F6368", fontsize=9)

    # Adapter (upper) and Base (lower). Title box then 0.3 gap then detail box.
    add_box(ax, 2.3, 5.3, BOX_W, 0.8, components["adapter_main"],
            fc=adapter_color, ec="#202124", weight="bold", fontsize=9)
    add_box(ax, 2.3, 3.4, BOX_W, LABEL_H, components["adapter_label"],
            fc=adapter_color, ec="#5F6368", fontsize=8, wrap_chars=20)

    add_box(ax, 2.3, 1.7, BOX_W, 0.8, components["base_main"],
            fc=base_color, ec="#202124", weight="bold", fontsize=9)
    # base detail sits below base title, leaving room for the caption at y=0.0
    # We skip a separate base_label box and put its info in the title box already.

    # Sum
    add_box(ax, 5.8, 3.0, 0.8, 0.8, "+",
            fc="#FFFFFF", ec="#5F6368", fontsize=14, weight="bold", lw=1.2)

    # Output
    add_box(ax, 7.1, 3.0, OUT_W, 0.8, "Output y",
            fc="#F1F3F4", ec="#5F6368", fontsize=9)

    # Caption strip at the very bottom of the subplot. 20% taller (0.7 -> 0.84)
    # so wrapped text has room. wrap_chars sized so text actually fits visually
    # inside the box at fontsize 7.5.
    add_box(ax, 0.3, 0.05, 11.4, 0.84, components["side_text"],
            fc=base_fc, ec="#CCCCCC", fontsize=7.5, wrap_chars=80)

    # Arrows
    add_arrow(ax, 1.7, 3.45, 2.3, 5.7)   # input to adapter
    add_arrow(ax, 1.7, 3.45, 2.3, 2.1)   # input to base
    add_arrow(ax, 2.3 + BOX_W, 5.7, 5.8, 3.6)   # adapter to sum
    add_arrow(ax, 2.3 + BOX_W, 2.1, 5.8, 3.2)   # base to sum
    add_arrow(ax, 6.6, 3.4, 7.1, 3.4)            # sum to output


# Wider figure, shorter subplot heights so vertical whitespace shrinks.
fig, axes = plt.subplots(3, 1, figsize=(13, 11))

make_method_panel(axes[0],
    title="(a) LoRA",
    components={
        "adapter_main": "Adapter",
        "adapter_label": "A and B in FP32, rank r=8, frozen base bypass",
        "base_main": "FP32 Conv",
        "side_text": ("Standard LoRA. Adapter A and B in FP32, rank r=8. Base conv FP32, frozen, no quantization. "
                      "Output y = base(x) + (B A)(x). At init B = 0 so y = base(x) exactly."),
    },
    base_color="#FCE8E6", adapter_color="#FCE8E6",
    quant_color="#FFFFFF", base_fc="#FFF5F4")

make_method_panel(axes[1],
    title="(b) QLoRA",
    components={
        "adapter_main": "Adapter",
        "adapter_label": "A and B in FP32, rank r=8, same as LoRA",
        "base_main": "NF4 Conv",
        "side_text": ("QLoRA (Dettmers 2023). Adapter: FP32 LoRA, same as LoRA. Base conv: 4-bit NF4 via bitsandbytes. "
                      "Dequant to BF16 each forward pass. Compute dtype BF16 (paper-mandated). Storage about 7.6 MB."),
    },
    base_color="#FEF7E0", adapter_color="#FEF7E0",
    quant_color="#FEF7E0", base_fc="#FFFBF0")

make_method_panel(axes[2],
    title="(c) QA-LoRA",
    components={
        "adapter_main": "Grouped Adapter",
        "adapter_label": "A shape L x r and B shape C_out x r, L=4 r=8",
        "base_main": "INT4 Conv",
        "side_text": ("QA-LoRA (Xu 2024) adapted to 2D convs. Adapter: grouped A shape L x r, standard B. "
                      "Base conv: group-wise INT4 quantization. Per-group learnable scale alpha_j & zero-point beta_j. "
                      "True-quant base, dequant with learned scale & zp in forward."),
    },
    base_color="#E6F4EA", adapter_color="#E6F4EA",
    quant_color="#E8F0FE", base_fc="#F1F8F4")

# tight_layout with rect to leave room for the bottom of each side panel
plt.tight_layout(rect=[0, 0, 1, 0.98])
out = os.path.join(OUT_DIR, "method_diagrams.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
