import os
import gdown
import zipfile
from pathlib import Path
from typing import Optional
import requests


class DatasetDownloader:
    """Download multimodal sentiment analysis datasets"""
    
    DATASETS = {
        "MOSI": {
            "url": "https://drive.google.com/uc?id=1woJZaKSVzx2Jc6ZM3sHFCGfxkxKu7aQZ",
            "filename": "MOSI.zip",
            "description": "Multimodal Opinion Sentiment Intensity"
        },
        "MOSEI": {
            "url": "https://drive.google.com/uc?id=1i2p_zd-I1M-xPrwhx3n-V4cI-5-9r5L1",
            "filename": "MOSEI.zip",
            "description": "Multimodal Opinion Sentiment and Emotion Intensity"
        }
    }
    
    def __init__(self, root_dir: str = "./data"):
        """Initialize downloader"""
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> Path:
        """
        Download dataset
        
        Args:
            dataset_name: Name of dataset (MOSI, MOSEI, etc.)
            force: Force download even if exists
            
        Returns:
            Path to extracted dataset
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_info = self.DATASETS[dataset_name]
        dataset_dir = self.root_dir / dataset_name
        
        # Check if already exists
        if dataset_dir.exists() and not force:
            print(f"Dataset {dataset_name} already exists at {dataset_dir}")
            return dataset_dir
        
        print(f"\nDownloading {dataset_name}...")
        print(f"Description: {dataset_info['description']}")
        
        # Download
        zip_path = self.root_dir / dataset_info['filename']
        
        try:
            gdown.download(dataset_info['url'], str(zip_path), quiet=False)
            
            # Extract
            print(f"\nExtracting {dataset_info['filename']}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(str(self.root_dir))
            
            # Clean up zip
            zip_path.unlink()
            
            print(f"✓ {dataset_name} downloaded successfully to {dataset_dir}")
            return dataset_dir
            
        except Exception as e:
            print(f"✗ Error downloading {dataset_name}: {str(e)}")
            raise
    
    def list_available_datasets(self) -> None:
        """List available datasets"""
        print("\nAvailable Datasets:")
        print("-" * 60)
        for name, info in self.DATASETS.items():
            print(f"  {name:15} - {info['description']}")
        print("-" * 60)


def create_sample_dataset(root_dir: str = "./data", dataset_name: str = "MOSI") -> Path:
    """
    Create a small sample dataset for testing
    
    Args:
        root_dir: Root directory for data
        dataset_name: Name of dataset
        
    Returns:
        Path to dataset
    """
    import json
    import numpy as np
    from pathlib import Path
    
    dataset_dir = Path(root_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data structure
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = dataset_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # Create sample metadata
        metadata = {
            'samples': []
        }
        
        for i in range(10):  # 10 samples per split
            sample = {
                'id': f'{split}_{i:04d}',
                'text': f"This is a sample sentiment text #{i}",
                'label': 1 if i % 2 == 0 else 0,
                'video_path': f'{split}_{i:04d}.mp4',
                'audio_path': f'{split}_{i:04d}.wav'
            }
            metadata['samples'].append(sample)
        
        # Save metadata
        with open(split_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create dummy video frames (as numpy arrays)
        frames_dir = split_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        
        for i in range(10):
            frame_data = np.random.rand(10, 224, 224, 3).astype(np.float32)
            np.save(frames_dir / f'{split}_{i:04d}.npy', frame_data)
        
        # Create dummy audio files (as numpy arrays)
        audio_dir = split_dir / 'audio'
        audio_dir.mkdir(exist_ok=True)
        
        for i in range(10):
            audio_data = np.random.rand(16000 * 3).astype(np.float32)  # 3 seconds at 16kHz
            np.save(audio_dir / f'{split}_{i:04d}.npy', audio_data)
    
    print(f"✓ Sample dataset created at {dataset_dir}")
    return dataset_dir
