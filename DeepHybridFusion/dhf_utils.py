import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path


class DHFTrainingUtils:
    """Utilities for training DHF model"""
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray,
                         targets: np.ndarray,
                         task: str = 'binary') -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            task: Task type (binary, multiclass)
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(targets, predictions)
        
        if task == 'binary':
            precision = precision_score(targets, predictions, zero_division=0)
            recall = recall_score(targets, predictions, zero_division=0)
            f1 = f1_score(targets, predictions, zero_division=0)
        else:
            precision = precision_score(targets, predictions, average='weighted', zero_division=0)
            recall = recall_score(targets, predictions, average='weighted', zero_division=0)
            f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def save_checkpoint(model: nn.Module,
                       optimizer,
                       epoch: int,
                       save_path: str,
                       best_metric: float = None):
        """Save model checkpoint"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'best_metric': best_metric
        }
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    @staticmethod
    def load_checkpoint(model: nn.Module,
                       optimizer,
                       checkpoint_path: str,
                       device: str = 'cpu') -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get('epoch', 0)


class DHFDataset(torch.utils.data.Dataset):
    """Custom dataset for DHF"""
    
    def __init__(self,
                 text_features: np.ndarray,
                 image_features: np.ndarray,
                 audio_features: np.ndarray,
                 labels: np.ndarray):
        """
        Initialize dataset
        
        Args:
            text_features: Text feature arrays
            image_features: Image feature arrays
            audio_features: Audio feature arrays
            labels: Label arrays
        """
        self.text_features = text_features
        self.image_features = image_features
        self.audio_features = audio_features
        self.labels = labels
        
        assert len(text_features) == len(image_features) == len(audio_features) == len(labels)
        self.length = len(labels)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get sample"""
        text = torch.from_numpy(self.text_features[idx]).float()
        image = torch.from_numpy(self.image_features[idx]).float()
        audio = torch.from_numpy(self.audio_features[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return text, image, audio, label


def create_dhf_dataloader(text_features: np.ndarray,
                         image_features: np.ndarray,
                         audio_features: np.ndarray,
                         labels: np.ndarray,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 4) -> torch.utils.data.DataLoader:
    """Create DHF dataloader"""
    dataset = DHFDataset(text_features, image_features, audio_features, labels)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

# Loss Function Implementation Notes
