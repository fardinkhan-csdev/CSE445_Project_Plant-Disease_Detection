import json
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


PLANTDOC_TO_PLANTVILLAGE_MAPPING = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf": "Potato___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Soybean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}


def normalize_label_name(label_name: str) -> str:
    value = label_name.strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return value.strip()


def segment_leaf(pil_image: Image.Image, padding: float = 0.15) -> Image.Image:
    """Remove background clutter using HSV + Otsu + GrabCut + morphology.

    Pipeline (fast path):
      1. Downscale to max 320px on longest edge for speed.
      2. Convert to HSV color space.
      3. Otsu thresholding on the Saturation channel to isolate leaf vs background.
      4. Morphological pre-cleanup of the Otsu mask.
      5. GrabCut with GMM refinement using the cleaned mask as initialization.
      6. Morphological open+close to snap edges and remove floating noise.
      7. Crop tightly around the leaf and paste onto a 255-white canvas.
      8. Resize back to original resolution.
    """
    orig_w, orig_h = pil_image.size
    max_side = 320
    scale = min(1.0, max_side / max(orig_w, orig_h))
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    small = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    img = np.array(small)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    sh, sw = img_bgr.shape[:2]

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Otsu thresholding on saturation (green leaves = high saturation)
    _, otsu_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_v = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine: high saturation AND not too dark
    combined = cv2.bitwise_and(otsu_mask, otsu_v)

    # Pre-cleanup
    kernel_pre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_pre)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_pre)

    # Build GrabCut mask from Otsu result
    mask = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[combined > 0] = cv2.GC_PR_FGD
    # Central margin as definite foreground
    margin = 0.15
    cx0, cy0 = int(sw * margin), int(sh * margin)
    cx1, cy1 = int(sw * (1.0 - margin)), int(sh * (1.0 - margin))
    mask[cy0:cy1, cx0:cx1] = cv2.GC_PR_FGD
    # Border = definite background
    mask[0, :] = cv2.GC_BGD
    mask[-1, :] = cv2.GC_BGD
    mask[:, 0] = cv2.GC_BGD
    mask[:, -1] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

    # Morphological filtering (snap edges + delete floating noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

    ys, xs = np.where(mask2 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return pil_image

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1

    # Minimum leaf size guard: if detection is too small, fallback to original
    if bw < 8 or bh < 8 or (bw * bh) < (sw * sh) * 0.05:
        return pil_image

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    cx0 = max(0, x0 - pad_x)
    cy0 = max(0, y0 - pad_y)
    cx1 = min(sw, x1 + pad_x)
    cy1 = min(sh, y1 + pad_y)

    cropped_small = img_bgr[cy0:cy1, cx0:cx1]
    cropped_mask = mask2[cy0:cy1, cx0:cx1]

    canvas = np.ones_like(cropped_small) * 255
    canvas = np.where(cropped_mask[:, :, np.newaxis].astype(bool), cropped_small, canvas).astype(np.uint8)

    # Resize back to original resolution
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(canvas_rgb)
    if scale < 1.0:
        result = result.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    return result


class PlantDocCOCODataset(Dataset):
    """Load PlantDoc images from COCO annotations and map them to PlantVillage labels."""

    def __init__(self, root_dir: str, split_name: str, class_to_idx: Dict[str, int], transform=None, apply_segmentation: bool = False):
        self.root_dir = root_dir
        self.split_name = split_name
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        self.apply_segmentation = apply_segmentation
        self.samples: List[Tuple[str, int]] = []

        annotation_path = os.path.join(root_dir, split_name, "_annotations.coco.json")
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"PlantDoc annotation file not found: {annotation_path}")

        with open(annotation_path, "r", encoding="utf-8") as handle:
            coco_data = json.load(handle)

        category_map = {category["id"]: category["name"] for category in coco_data.get("categories", [])}
        annotations_by_image = {}
        for annotation in coco_data.get("annotations", []):
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

        for image_info in coco_data.get("images", []):
            image_id = image_info["id"]
            image_path = os.path.join(root_dir, split_name, image_info.get("file_name", ""))
            if not os.path.exists(image_path):
                continue

            matched_label = self._resolve_label(annotations_by_image.get(image_id, []), category_map)
            if matched_label is None:
                continue

            if matched_label not in self.class_to_idx:
                continue

            self.samples.append((image_path, self.class_to_idx[matched_label]))

    def _resolve_label(self, annotations: List[Dict], category_map: Dict[int, str]) -> Optional[str]:
        if not annotations:
            return None

        candidate_annotations = []
        for annotation in annotations:
            category_name = category_map.get(annotation.get("category_id"))
            if not category_name:
                continue
            mapped_label = PLANTDOC_TO_PLANTVILLAGE_MAPPING.get(category_name)
            if mapped_label:
                candidate_annotations.append((annotation, mapped_label))

        if not candidate_annotations:
            return None

        candidate_annotations.sort(key=lambda item: item[0].get("area", 0), reverse=True)
        return candidate_annotations[0][1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label_idx = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.apply_segmentation:
            image = segment_leaf(image)

        if self.transform:
            image = self.transform(image)

        label_name = self.idx_to_class.get(label_idx, str(label_idx))
        if "___" in label_name:
            crop, disease = label_name.split("___", 1)
        else:
            crop, disease = label_name, "healthy"

        return image, label_idx, crop, disease, image_path


def get_available_plantdoc_splits(root_dir: str) -> List[str]:
    available = []
    for split_name in ["train", "valid", "test"]:
        if os.path.isdir(os.path.join(root_dir, split_name)):
            available.append(split_name)
    return available


def load_class_labels(class_labels_path: Optional[str] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    if class_labels_path is None:
        class_labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "class_labels.json")
    with open(class_labels_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["class_to_idx"], {int(idx): label for idx, label in data["idx_to_class"].items()}
