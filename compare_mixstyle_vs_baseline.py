"""
Comparison script: MixStyle vs Baseline LoRA training

Trains two LoRA models on PlantVillage:
1. Baseline (no MixStyle)
2. MixStyle (with MixStyle enabled)

Then runs PlantDoc evaluation on both to compare cross-domain performance.
"""

import os
import sys
import shutil
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def backup_config(path):
    backup_path = path + '.backup'
    shutil.copy2(path, backup_path)
    return backup_path


def restore_config(path, backup_path):
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, path)
        os.remove(backup_path)


def modify_config_for_mixstyle(config_path, enable=True):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['training']['use_mixstyle'] = enable
    if enable:
        config['training']['mixstyle_prob'] = 0.5
        config['training']['mixstyle_layers'] = [0, 2, 4]
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def main():
    config_path = os.path.join(PROJECT_ROOT, 'config', 'base_config_v2.yaml')
    backup_path = backup_config(config_path)
    
    try:
        print("\n" + "=" * 70)
        print("MixStyle vs Baseline Comparison")
        print("=" * 70)
        
        print("\n[1/2] Training Baseline LoRA (no MixStyle)...")
        modify_config_for_mixstyle(config_path, enable=False)
        os.system(f'"{sys.executable}" main_v2.py lora')
        
        print("\n[2/2] Training LoRA with MixStyle...")
        modify_config_for_mixstyle(config_path, enable=True)
        os.system(f'"{sys.executable}" main_v2.py lora')
        
        print("\nDone! Check experiments/results/checkpoints_v2 for both checkpoints.")
        print("Run python launcher_plantdoc.py and choose option 8 (AdaBN) to evaluate both on PlantDoc.")
        
    finally:
        restore_config(config_path, backup_path)
        print("\nConfig restored to original state.")


if __name__ == "__main__":
    main()
