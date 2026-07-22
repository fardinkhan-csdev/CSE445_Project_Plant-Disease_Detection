import json
import os
import re
from typing import Dict, List, Optional, Tuple

from PIL import Image
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


class PlantDocCOCODataset(Dataset):
    """Load PlantDoc images from COCO annotations and map them to PlantVillage labels."""

    def __init__(self, root_dir: str, split_name: str, class_to_idx: Dict[str, int], transform=None):
        self.root_dir = root_dir
        self.split_name = split_name
        self.transform = transform
        self.class_to_idx = class_to_idx
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
        if self.transform:
            image = self.transform(image)
        return image, label_idx


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
