"""Generate aggregate metric comparison bar chart for the three PEFT methods."""
import os
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

methods = ["LoRA", "QLoRA", "QA-LoRA"]
accuracy = [98.89, 98.91, 99.52]
f1_macro = [98.36, 98.39, 99.33]
colors = ["#4285F4", "#FBBC04", "#34A853"]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))
bars1 = ax.bar(x - width / 2, accuracy, width, label="Test Accuracy", color="#34A853", edgecolor="black", linewidth=0.8)
bars2 = ax.bar(x + width / 2, f1_macro, width, label="Macro F1", color="#4285F4", edgecolor="black", linewidth=0.8)

for bars in (bars1, bars2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9, weight="bold")

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12, weight="bold")
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_title("Test accuracy and macro-F1 on PlantVillage", fontsize=12, weight="bold")
ax.set_ylim(98.0, 99.9)
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax.grid(True, axis="y", alpha=0.3, linestyle="--")

# Reference line for full-FT cited accuracy
ax.axhline(99.56, color="#EA4335", linestyle=":", alpha=0.5, linewidth=1.2)
ax.text(0.02, 99.585, "Full FT cited (99.56%)", fontsize=8, color="#EA4335", ha="left", transform=ax.get_yaxis_transform())

plt.tight_layout()
out = os.path.join(OUT_DIR, "metric_comparison.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
