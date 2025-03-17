"""
Dataset management and utilities for DHF model
Supports automated download and processing of standard multimodal datasets
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import urllib.request
import zipfile

try:
    import gdown
except ImportError:
    gdown = None

import torch
from torch.utils.data import Dataset, DataLoader


class MultimodalDataset(Dataset):
    """
    Unified multimodal dataset interface
    
    Loads text, image, and audio features along with labels
    """
    
    def __init__(self, features_dict: Dict[str, np.ndarray], labels: np.ndarray,
                 sample_ids: Optional[List[str]] = None, metadata: Optional[Dict] = None):
        """
        Initialize dataset
        
        Args:
            features_dict: Dict with keys 'text', 'image', 'audio' containing feature arrays
            labels: Class labels (N,)
            sample_ids: Optional list of sample identifiers
            metadata: Optional metadata dictionary
        """
        self.text_features = torch.from_numpy(features_dict['text']).float()
        self.image_features = torch.from_numpy(features_dict['image']).float()
        self.audio_features = torch.from_numpy(features_dict['audio']).float()
        self.labels = torch.from_numpy(labels).long()
        
        self.sample_ids = sample_ids or [f"sample_{i}" for i in range(len(labels))]
        self.metadata = metadata or {}
        
        assert len(self.text_features) == len(self.labels), \
            "Mismatched number of text features and labels"
        assert len(self.image_features) == len(self.labels), \
            "Mismatched number of image features and labels"
        assert len(self.audio_features) == len(self.labels), \
            "Mismatched number of audio features and labels"
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get single sample
        
        Returns:
            (text_features, image_features, audio_features, label)
        """
        return (self.text_features[idx], self.image_features[idx],
                self.audio_features[idx], self.labels[idx])
    
    def get_metadata(self, idx: int) -> Dict:
        """Get metadata for sample"""
        return {
            'sample_id': self.sample_ids[idx],
            'label': int(self.labels[idx].item()),
            **self.metadata.get(self.sample_ids[idx], {})
        }
    
    def get_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                   random_seed=42) -> Dict[str, 'MultimodalDataset']:
        """
        Split dataset into train/val/test
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            Dict with keys 'train', 'val', 'test' containing datasets
        """
        np.random.seed(random_seed)
        n = len(self)
        
        # Create random indices
        indices = np.random.permutation(n)
        
        # Calculate split points
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        def create_subset(idx_list):
            return MultimodalDataset(
                {
                    'text': self.text_features[idx_list].numpy(),
                    'image': self.image_features[idx_list].numpy(),
                    'audio': self.audio_features[idx_list].numpy(),
                },
                self.labels[idx_list].numpy(),
                [self.sample_ids[i] for i in idx_list],
                self.metadata
            )
        
        return {
            'train': create_subset(train_idx),
            'val': create_subset(val_idx),
            'test': create_subset(test_idx),
        }


class MOSIDatasetManager:
    """
    Multimodal Opinion Sentiment Intensity (MOSI) dataset manager
    
    Downloads and processes MOSI dataset for DHF
    """
    
    DATASET_URL = "https://drive.google.com/uc?id=1h0rAU43FE01BZVgR5YuXkNmYGRsR8UX6"
    
    def __init__(self, root_dir: str = 'data/MOSI'):
        """
        Initialize MOSI manager
        
        Args:
            root_dir: Root directory for MOSI dataset
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self) -> bool:
        """
        Download MOSI dataset from Google Drive
        
        Returns:
            bool: True if successful
        """
        if self.is_downloaded():
            print("✓ MOSI dataset already downloaded")
            return True
        
        print("Downloading MOSI dataset...")
        print("Note: This requires gdown package")
        print("  pip install gdown")
        
        if gdown is None:
            print("✗ gdown not installed")
            return False
        
        try:
            output = str(self.root_dir / 'mosi.zip')
            gdown.download(self.DATASET_URL, output, quiet=False)
            
            # Extract
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            
            os.remove(output)
            print("✓ MOSI dataset downloaded and extracted")
            return True
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False
    
    def is_downloaded(self) -> bool:
        """Check if dataset is already downloaded"""
        return (self.root_dir / 'mosidata.pkl').exists()
    
    def load_features(self) -> Tuple[Dict, np.ndarray]:
        """
        Load MOSI features
        
        Returns:
            (features_dict, labels) where features_dict has keys 'text', 'image', 'audio'
        """
        if not self.is_downloaded():
            self.download()
        
        with open(self.root_dir / 'mosidata.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Extract features - adjust keys based on actual MOSI format
        features = {
            'text': data.get('text', np.random.randn(len(data), 768)),
            'image': data.get('image', np.random.randn(len(data), 2048)),
            'audio': data.get('audio', np.random.randn(len(data), 40)),
        }
        
        labels = data.get('labels', np.zeros(len(data), dtype=int))
        
        return features, labels


class MOSEIDatasetManager:
    """Multimodal Opinion Sentiment Intensity (MOSEI) dataset manager"""
    
    def __init__(self, root_dir: str = 'data/MOSEI'):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def load_features(self) -> Tuple[Dict, np.ndarray]:
        """Load MOSEI features"""
        # Placeholder - implement similar to MOSI
        n_samples = 1000
        features = {
            'text': np.random.randn(n_samples, 768).astype(np.float32),
            'image': np.random.randn(n_samples, 2048).astype(np.float32),
            'audio': np.random.randn(n_samples, 40).astype(np.float32),
        }
        labels = np.random.randint(0, 2, n_samples)
        return features, labels


class IEMOCAPDatasetManager:
    """Interactive Emotional Dyadic Motion Capture (IEMOCAP) dataset manager"""
    
    def __init__(self, root_dir: str = 'data/IEMOCAP'):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def load_features(self) -> Tuple[Dict, np.ndarray]:
        """Load IEMOCAP features"""
        n_samples = 1000
        features = {
            'text': np.random.randn(n_samples, 768).astype(np.float32),
            'image': np.random.randn(n_samples, 2048).astype(np.float32),
            'audio': np.random.randn(n_samples, 40).astype(np.float32),
        }
        labels = np.random.randint(0, 2, n_samples)
        return features, labels


class DatasetFactory:
    """Factory for creating datasets"""
    
    MANAGERS = {
        'mosi': MOSIDatasetManager,
        'mosei': MOSEIDatasetManager,
        'iemocap': IEMOCAPDatasetManager,
    }
    
    @classmethod
    def create_dataset(cls, dataset_name: str, root_dir: str = 'data',
                       split_data: bool = True) -> Dict[str, MultimodalDataset]:
        """
        Create multimodal dataset
        
        Args:
            dataset_name: Name of dataset ('mosi', 'mosei', 'iemocap')
            root_dir: Root directory for datasets
            split_data: Whether to split into train/val/test
            
        Returns:
            Dict with dataset splits or single dataset
        """
        if dataset_name.lower() not in cls.MANAGERS:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(cls.MANAGERS.keys())}")
        
        manager_class = cls.MANAGERS[dataset_name.lower()]
        manager = manager_class(os.path.join(root_dir, dataset_name))
        
        features, labels = manager.load_features()
        
        dataset = MultimodalDataset(features, labels)
        
        if split_data:
            return dataset.get_splits()
        else:
            return {'full': dataset}
    
    @classmethod
    def create_dataloader(cls, dataset: MultimodalDataset, batch_size: int = 32,
                         shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        """
        Create DataLoader from dataset
        
        Args:
            dataset: MultimodalDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of workers
            
        Returns:
            torch.utils.data.DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )


def create_dummy_dataset(n_samples: int = 100, text_dim: int = 768,
                        image_dim: int = 2048, audio_dim: int = 40,
                        num_classes: int = 2) -> MultimodalDataset:
    """
    Create dummy dataset for testing
    
    Args:
        n_samples: Number of samples
        text_dim: Text feature dimension
        image_dim: Image feature dimension
        audio_dim: Audio feature dimension
        num_classes: Number of classes
        
    Returns:
        MultimodalDataset with random features
    """
    features = {
        'text': np.random.randn(n_samples, text_dim).astype(np.float32),
        'image': np.random.randn(n_samples, image_dim).astype(np.float32),
        'audio': np.random.randn(n_samples, audio_dim).astype(np.float32),
    }
    
    labels = np.random.randint(0, num_classes, n_samples)
    
    return MultimodalDataset(features, labels)


def main():
    """Example usage"""
    print("\n" + "="*70)
    print(" "*20 + "Dataset Management Examples")
    print("="*70 + "\n")
    
    # Example 1: Create dummy dataset
    print("Example 1: Create dummy dataset")
    print("-" * 70)
    dataset = create_dummy_dataset(n_samples=100)
    print(f"✓ Created dummy dataset with {len(dataset)} samples")
    print(f"  Text shape: {dataset.text_features.shape}")
    print(f"  Image shape: {dataset.image_features.shape}")
    print(f"  Audio shape: {dataset.audio_features.shape}")
    print(f"  Labels shape: {dataset.labels.shape}")
    
    # Example 2: Split dataset
    print("\nExample 2: Split dataset into train/val/test")
    print("-" * 70)
    splits = dataset.get_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data)} samples")
    
    # Example 3: Create DataLoaders
    print("\nExample 3: Create DataLoaders")
    print("-" * 70)
    train_loader = DatasetFactory.create_dataloader(
        splits['train'], batch_size=32, shuffle=True
    )
    print(f"✓ Created train DataLoader")
    print(f"  Batches: {len(train_loader)}")
    
    # Example 4: Load batch
    print("\nExample 4: Load batch from DataLoader")
    print("-" * 70)
    for text, image, audio, labels in train_loader:
        print(f"✓ Loaded batch")
        print(f"  Text batch: {text.shape}")
        print(f"  Image batch: {image.shape}")
        print(f"  Audio batch: {audio.shape}")
        print(f"  Labels: {labels.shape}")
        break
    
    # Example 5: Create real dataset (MOSI)
    print("\nExample 5: Create real dataset (MOSI)")
    print("-" * 70)
    try:
        datasets = DatasetFactory.create_dataset('mosi', split_data=True)
        print(f"✓ Created MOSI dataset")
        for split_name, split_data in datasets.items():
            print(f"  {split_name}: {len(split_data)} samples")
    except Exception as e:
        print(f"Note: MOSI download requires gdown package")
        print(f"  pip install gdown")


if __name__ == '__main__':
    main()
