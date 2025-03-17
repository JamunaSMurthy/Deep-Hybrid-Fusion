import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DHFTrainer:
    """Trainer class for DHF model"""
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize trainer"""
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.0001),
            weight_decay=config.get('weight_decay', 0.0001)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('epochs', 50)
            )
        elif scheduler_type == 'linear':
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                total_iters=config.get('epochs', 50)
            )
        else:
            self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Best metrics tracking
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for text, image, audio, labels in pbar:
            # Move to device
            text = text.to(self.device)
            image = image.to(self.device)
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(text, image, audio)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('gradient_clip', 1.0))
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': correct / total
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc="Validating")
        
        for text, image, audio, labels in pbar:
            # Move to device
            text = text.to(self.device)
            image = image.to(self.device)
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits, _ = self.model(text, image, audio)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': correct / total
            })
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = None,
            save_dir: str = './models') -> Dict:
        """Train model"""
        epochs = epochs or self.config.get('epochs', 50)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        early_stopping = self.config.get('early_stopping', True)
        patience = self.config.get('patience', 10)
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Early stopping
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                
                # Save best model
                checkpoint_path = save_dir / f'best_model.pt'
                self._save_checkpoint(checkpoint_path, epoch)
                print(f"✓ Best model saved (Acc: {self.best_val_acc:.4f})")
            else:
                self.patience_counter += 1
                
                if early_stopping and self.patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Save final model
        final_checkpoint = save_dir / f'final_model.pt'
        self._save_checkpoint(final_checkpoint, epochs)
        
        # Save training history
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n✓ Training completed!")
        print(f"✓ Models saved in {save_dir}")
        
        return self.history
    
    def _save_checkpoint(self, path: Path, epoch: int):
        """Save checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        torch.save(checkpoint, path)


class DHFInference:
    """Inference utilities for DHF model"""
    
    def __init__(self, model: nn.Module, checkpoint_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize inference"""
        self.device = device
        self.model = model.to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    @torch.no_grad()
    def predict(self,
                text_features: np.ndarray,
                image_features: np.ndarray,
                audio_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            text_features: Text features
            image_features: Image features
            audio_features: Audio features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # Convert to tensors
        text = torch.from_numpy(text_features).float().to(self.device)
        image = torch.from_numpy(image_features).float().to(self.device)
        audio = torch.from_numpy(audio_features).float().to(self.device)
        
        # Forward pass
        logits, _ = self.model(text, image, audio)
        
        # Get predictions
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy(), probs.cpu().numpy()
    
    @torch.no_grad()
    def predict_batch(self,
                     text_batch: List[np.ndarray],
                     image_batch: List[np.ndarray],
                     audio_batch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict for batch"""
        text = torch.from_numpy(np.array(text_batch)).float().to(self.device)
        image = torch.from_numpy(np.array(image_batch)).float().to(self.device)
        audio = torch.from_numpy(np.array(audio_batch)).float().to(self.device)
        
        logits, _ = self.model(text, image, audio)
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy(), probs.cpu().numpy()
