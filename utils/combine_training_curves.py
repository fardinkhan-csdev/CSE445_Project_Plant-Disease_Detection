"""Combine LoRA, QLoRA, and QA-LoRA training curves into one 3-column figure."""
from PIL import Image
import os

SRC_DIR = "paper_figures"
OUT_DIR = "paper_submission"
os.makedirs(OUT_DIR, exist_ok=True)

files = [
    ("lora_training_curves.png", "(a) LoRA"),
    ("qlora_training_curves.png", "(b) QLoRA"),
    ("qalora_training_curves.png", "(c) QA-LoRA"),
]

# Load and add column labels by compositing onto a top strip.
imgs = []
for fname, label in files:
    path = os.path.join(SRC_DIR, fname)
    img = Image.open(path).convert("RGB")
    imgs.append((img, label))

# Use the first image dimensions as reference; all three should match.
ref_w, ref_h = imgs[0][0].size
label_height = max(40, ref_h // 20)  # 5% of height for label strip

# Resize all to same size (they may already match).
resized = []
for img, label in imgs:
    if img.size != (ref_w, ref_h):
        img = img.resize((ref_w, ref_h), Image.LANCZOS)
    resized.append((img, label))

# Build final image: label strip on top, then 3 columns side by side.
final_w = ref_w * 3
final_h = ref_h + label_height
final = Image.new("RGB", (final_w, final_h), color=(255, 255, 255))

# Paste label strip at top.
# We'll draw labels using PIL's ImageDraw if available, else skip.
try:
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(final)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, label_height // 3))
    except Exception:
        font = ImageFont.load_default()
    col_w = ref_w
    for i, (_, label) in enumerate(resized):
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = i * col_w + (col_w - tw) // 2
        y = (label_height - th) // 2
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
except Exception:
    pass

# Paste images below the label strip.
for i, (img, _) in enumerate(resized):
    final.paste(img, (i * ref_w, label_height))

out_path = os.path.join(OUT_DIR, "all_training_curves.png")
final.save(out_path, dpi=(200, 200))
print(f"Saved: {out_path} ({final.size})")
