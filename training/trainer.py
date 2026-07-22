import os
import glob
import time
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.visualization import plot_training_curves
from utils.memory_tracker import get_gpu_memory_stats, reset_gpu_memory_stats
from evaluation.evaluator import Evaluator


def prepare_model_state_dict_for_load(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop legacy duplicate FP32 base weights from older QLoRA/QKLoRA checkpoints."""
    int8_prefixes = {k[: -len('.weight_int8')] for k in state_dict if k.endswith('.weight_int8')}
    if not int8_prefixes:
        return state_dict
    return {
        k: v
        for k, v in state_dict.items()
        if not (k.endswith('.base_layer.weight') and k[: -len('.weight')] in int8_prefixes)
    }


class BaseTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 config_path: str = 'config/base_config.yaml', experiment_name: str = 'base',
                 class_weights: Optional[torch.Tensor] = None):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_name = experiment_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set device
        if torch.cuda.is_available():
            try:
                self.device = torch.device('cuda')
                self.model.to(self.device)
                torch.zeros(1, device=self.device)
            except Exception:
                self.device = torch.device('cpu')
                self.model.to(self.device)
        else:
            self.device = torch.device('cpu')
            self.model.to(self.device)
        
        speed_config = self.config.get('speed', {})
        if self.device.type == 'cuda':
            if speed_config.get('cudnn_benchmark', True):
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if speed_config.get('compile', False):
                self.model = torch.compile(self.model, mode='max-autotune')
        
        self.use_amp = bool(speed_config.get('mixed_precision', True)) and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        
        # Training config
        training_config = self.config['training']
        self.epochs = int(training_config['epochs'])
        self.lr = float(training_config['learning_rate'])
        self.weight_decay = float(training_config['weight_decay'])
        self.early_stopping_patience = int(training_config.get('early_stopping_patience', 3))
        
        # Logging config
        logging_config = self.config['logging']
        self.log_dir = logging_config['log_dir']
        self.checkpoint_dir = logging_config['checkpoint_dir']
        self.plot_dir = logging_config['plot_dir']
        self.save_epoch_checkpoints = bool(logging_config.get('save_epoch_checkpoints', True))
        
        # Setup logger
        self.logger = setup_logger(self.log_dir, name=experiment_name)
        
        # Setup optimizer, scheduler, loss (driven by config/base_config.yaml)
        self.optimizer = self._build_optimizer(training_config)
        self.scheduler = self._build_scheduler(training_config)
        self.criterion = self._build_criterion(training_config, class_weights)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.training_time = 0.0
        self.peak_gpu_memory = 0.0

    def _build_optimizer(self, training_config: Dict) -> optim.Optimizer:
        optimizer_name = training_config.get('optimizer', 'AdamW')
        if optimizer_name == 'AdamW':
            return optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if optimizer_name == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _build_scheduler(self, training_config: Dict):
        scheduler_name = training_config.get('scheduler', 'CosineAnnealingLR')
        if scheduler_name == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def _build_criterion(self, training_config: Dict, class_weights: Optional[torch.Tensor]) -> nn.Module:
        loss_name = training_config.get('loss_fn', 'CrossEntropyLoss')
        if loss_name == 'CrossEntropyLoss':
            weight = class_weights.to(self.device) if class_weights is not None else None
            return nn.CrossEntropyLoss(weight=weight)
        raise ValueError(f"Unsupported loss function: {loss_name}")
    
    def train_epoch(self) -> tuple:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # Dataset now returns (image, label, crop, disease, image_path)
            images, labels = batch[0], batch[1]
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def val_epoch(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                images, labels = batch[0], batch[1]
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def _update_status(self, status: str, current_epoch: int, patience_counter: int):
        status_dir = os.path.dirname(self.checkpoint_dir)
        os.makedirs(status_dir, exist_ok=True)
        status_path = os.path.join(status_dir, 'training_status.json')
        status_data = {
            'experiment_name': self.experiment_name,
            'status': status,
            'current_epoch': current_epoch,
            'total_epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'patience_counter': patience_counter
        }
        try:
            with open(status_path, 'w') as f:
                json.dump(status_data, f, indent=4)
        except Exception as e:
            self.logger.warning(f"Failed to write training status: {e}")

    def train(self, evaluator: Optional[Evaluator] = None, resume: bool = False):
        self.logger.info(f"Starting training for {self.experiment_name}")
        self.logger.info(f"Device: {self.device}")
        reset_gpu_memory_stats()
        self.start_time = time.time()
        
        self.patience_counter = 0
        start_epoch = 0
        
        if resume:
            checkpoint_path, _ = self._build_checkpoint_paths('latest')
            if os.path.exists(checkpoint_path):
                self.load_checkpoint('latest')
                start_epoch = len(self.train_losses)
                self.patience_counter = getattr(self, 'loaded_patience_counter', 0)
                self.logger.info(f"Resuming training from epoch {start_epoch + 1}")
                # Adjust start time to account for already spent training time
                self.start_time = time.time() - self.training_time
            else:
                self.logger.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        
        self._update_status('running', start_epoch, self.patience_counter)
        
        for epoch in range(start_epoch, self.epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.val_epoch()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Update dynamic variables before checkpoint saving
            self.training_time = time.time() - self.start_time
            if torch.cuda.is_available():
                gpu_stats = get_gpu_memory_stats()
                self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_stats['max_allocated'])

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best')
                self.logger.info(f"Best model saved with Val Acc: {val_acc:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                self.logger.info(f"No improvement for {self.patience_counter}/{self.early_stopping_patience} epochs")
            
            # Save latest checkpoint at the end of every epoch regardless of config
            self.save_checkpoint('latest')
            self._update_status('running', epoch + 1, self.patience_counter)
            
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                break

            # Optionally save epoch checkpoint
            if self.save_epoch_checkpoints:
                self.save_checkpoint(f'epoch_{epoch+1}')
            
            # Track and log GPU memory
            if torch.cuda.is_available():
                gpu_stats = get_gpu_memory_stats()
                self.logger.info(f"GPU Memory: {gpu_stats['allocated']:.2f} GB (Peak: {self.peak_gpu_memory:.2f} GB)")
            else:
                self.logger.info(f"GPU Memory: 0.00 GB (Peak: 0.00 GB)")
        
        # Calculate total training time and peak GPU memory one final time
        self.training_time = time.time() - self.start_time
        if torch.cuda.is_available():
            gpu_stats = get_gpu_memory_stats()
            self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_stats['max_allocated'])

        # Save last model
        self.save_checkpoint('last')
        
        self.logger.info(f"\nTraining completed in {self.training_time:.2f} seconds")
        self.logger.info(f"Best Val Acc: {self.best_val_acc:.4f}")
        self.logger.info(f"Peak GPU Memory: {self.peak_gpu_memory:.2f} GB")
        
        # Mark status as completed
        self._update_status('completed', len(self.train_losses), self.patience_counter)
        
        # Plot training curves
        plot_training_curves(self.train_losses, self.val_losses, self.train_accs, self.val_accs, 
                            self.plot_dir, self.experiment_name, run_tag=self.run_id)
    
    def _build_checkpoint_paths(self, name: str):
        base_name = f'{self.experiment_name}_{name}'
        primary_path = os.path.join(self.checkpoint_dir, f'{base_name}.pth')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup_path = os.path.join(self.checkpoint_dir, f'{base_name}_{timestamp}.pth')
        return primary_path, backup_path
    
    def save_checkpoint(self, name: str):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_payload = {
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'training_time': self.training_time,
            'peak_gpu_memory': self.peak_gpu_memory,
            'patience_counter': getattr(self, 'patience_counter', 0)
        }
        primary_path, backup_path = self._build_checkpoint_paths(name)
        torch.save(checkpoint_payload, primary_path)
        torch.save(checkpoint_payload, backup_path)
        self.logger.info(f"Checkpoint saved to {primary_path}")
        self.logger.info(f"Checkpoint backup saved to {backup_path}")
    
    def load_checkpoint(self, name: str):
        primary_path, _ = self._build_checkpoint_paths(name)
        candidate_paths = [primary_path]
        candidate_paths.extend(sorted(glob.glob(os.path.join(self.checkpoint_dir, f'{self.experiment_name}_{name}_*.pth'))))
        checkpoint_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = prepare_model_state_dict_for_load(checkpoint['model_state_dict'])
            load_result = self.model.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                self.logger.warning(f"Checkpoint missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                self.logger.warning(f"Checkpoint unexpected keys: {load_result.unexpected_keys}")
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accs = checkpoint['train_accs']
            self.val_accs = checkpoint['val_accs']
            self.best_val_acc = checkpoint['best_val_acc']
            self.training_time = max(self.training_time, checkpoint.get('training_time', 0.0))
            self.peak_gpu_memory = max(self.peak_gpu_memory, checkpoint.get('peak_gpu_memory', 0.0))
            self.loaded_patience_counter = checkpoint.get('patience_counter', 0)
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        else:
            self.logger.warning(f"Checkpoint not found at {checkpoint_path}")
