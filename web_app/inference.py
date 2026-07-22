import os
import sys
import time
import json
import torch
import torchvision.transforms as T
from PIL import Image
import io

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.backbone.efficientnet_b0 import get_efficientnet_b0
from models.classifier import PlantDiseaseClassifier
from training.lora_trainer import LoRATrainer
from training.qlora_trainer import QLoRATrainer
from training.qklora_trainer import QKLoRATrainer
from training.trainer import prepare_model_state_dict_for_load

class InferenceEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels_path = os.path.join(PROJECT_ROOT, 'config', 'class_labels.json')
        self.load_class_labels()
        self.models = {}
        
        # Standard ImageNet val/test evaluation pipeline
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_class_labels(self):
        if os.path.exists(self.class_labels_path):
            with open(self.class_labels_path, 'r') as f:
                data = json.load(f)
                self.idx_to_class = {int(k): v for k, v in data.get('idx_to_class', {}).items()}
                self.num_classes = data.get('num_classes', len(self.idx_to_class))
        else:
            # Fallback if class_labels.json doesn't exist yet
            self.idx_to_class = {i: f"Class_{i}" for i in range(38)}
            self.num_classes = 38

    def get_model(self, model_key: str):
        model_key = model_key.lower().strip()
        if model_key in self.models:
            return self.models[model_key]

        checkpoint_dir = os.path.join(PROJECT_ROOT, 'experiments', 'results', 'checkpoints')
        possible_files = [
            f"{model_key}_best.pth",
            f"{model_key}_last.pth",
            f"{model_key}_latest.pth"
        ]
        
        ckpt_path = None
        for fn in possible_files:
            fp = os.path.join(checkpoint_dir, fn)
            if os.path.exists(fp):
                ckpt_path = fp
                break
        
        if not ckpt_path:
            # Fallback search any matching .pth
            if os.path.exists(checkpoint_dir):
                for f in os.listdir(checkpoint_dir):
                    if f.startswith(model_key) and f.endswith('.pth'):
                        ckpt_path = os.path.join(checkpoint_dir, f)
                        break

        # Instantiate trainer to build exact architecture with PEFT modules
        trainer_map = {
            'lora': LoRATrainer,
            'qlora': QLoRATrainer,
            'qklora': QKLoRATrainer
        }
        
        trainer_cls = trainer_map.get(model_key, LoRATrainer)
        # Dummy data parameters just for model initialization
        trainer = trainer_cls(None, None, num_classes=self.num_classes)
        model = trainer.model.to(self.device)

        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            state_dict = prepare_model_state_dict_for_load(state_dict)
            model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded checkpoint for {model_key}: {ckpt_path}")
        else:
            print(f"Warning: No checkpoint found for {model_key} at {checkpoint_dir}. Using initial weights.")

        model.eval()
        self.models[model_key] = model
        return model

    def predict(self, image_input, model_key='lora', top_k=5):
        start_time = time.time()
        
        if isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            img = Image.open(image_input).convert('RGB')

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        model = self.get_model(model_key)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        top_probs, top_indices = torch.topk(probs, min(top_k, self.num_classes))
        
        top_k_list = []
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            cls_name = self.idx_to_class.get(idx, f"Unknown_{idx}")
            top_k_list.append({
                "class_name": cls_name,
                "confidence": round(p * 100, 2),
                "prob": float(p)
            })

        best_pred = top_k_list[0]
        full_class = best_pred["class_name"]
        
        if "___" in full_class:
            crop, disease = full_class.split("___", 1)
        else:
            crop = full_class
            disease = "unknown"

        is_healthy = "healthy" in disease.lower()
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "model_used": model_key.upper(),
            "predicted_class": full_class,
            "crop": crop.replace("_", " ").title(),
            "disease": disease.replace("_", " ").title(),
            "is_healthy": is_healthy,
            "status": "Healthy" if is_healthy else "Diseased",
            "confidence": best_pred["confidence"],
            "latency_ms": latency_ms,
            "top_k": top_k_list
        }

# Global singleton engine instance
engine = InferenceEngine()
