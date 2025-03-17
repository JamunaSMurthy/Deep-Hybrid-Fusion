import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2


class DataPreprocessor:
    """Preprocess multimodal data"""
    
    @staticmethod
    def normalize_features(features: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        Normalize features
        
        Args:
            features: Feature array of shape (N, D)
            method: Normalization method ('standard', 'minmax', 'l2')
            
        Returns:
            Normalized features
        """
        if method == 'standard':
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True) + 1e-8
            return (features - mean) / std
        
        elif method == 'minmax':
            min_val = features.min(axis=0, keepdims=True)
            max_val = features.max(axis=0, keepdims=True)
            return (features - min_val) / (max_val - min_val + 1e-8)
        
        elif method == 'l2':
            norm = np.linalg.norm(features, axis=1, keepdims=True)
            return features / (norm + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def pad_or_truncate(data: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
        """
        Pad or truncate sequence to target length
        
        Args:
            data: Input array
            target_length: Target length
            axis: Axis along which to pad/truncate
            
        Returns:
            Padded/truncated array
        """
        current_length = data.shape[axis]
        
        if current_length == target_length:
            return data
        
        elif current_length < target_length:
            # Pad with zeros
            pad_width = [(0, 0)] * data.ndim
            pad_width[axis] = (0, target_length - current_length)
            return np.pad(data, pad_width, mode='constant', constant_values=0)
        
        else:
            # Truncate
            slices = [slice(None)] * data.ndim
            slices[axis] = slice(0, target_length)
            return data[tuple(slices)]
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image (H, W, C)
            target_size: Target size (H, W)
            
        Returns:
            Resized image
        """
        return cv2.resize(image, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def align_modalities(text_features: np.ndarray,
                        image_features: np.ndarray,
                        audio_features: np.ndarray,
                        target_length: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align different modalities to same temporal length
        
        Args:
            text_features: Text features (L_t, D_t)
            image_features: Image features (L_i, D_i)
            audio_features: Audio features (L_a, D_a)
            target_length: Target sequence length
            
        Returns:
            Aligned features for each modality
        """
        text_aligned = DataPreprocessor.pad_or_truncate(text_features, target_length, axis=0)
        image_aligned = DataPreprocessor.pad_or_truncate(image_features, target_length, axis=0)
        audio_aligned = DataPreprocessor.pad_or_truncate(audio_features, target_length, axis=0)
        
        return text_aligned, image_aligned, audio_aligned
    
    @staticmethod
    def create_attention_mask(length: int, actual_length: int) -> np.ndarray:
        """
        Create attention mask for padded sequences
        
        Args:
            length: Total padded length
            actual_length: Actual sequence length
            
        Returns:
            Boolean mask (True for valid, False for padding)
        """
        mask = np.ones(length, dtype=np.bool_)
        mask[actual_length:] = False
        return mask


class MetadataProcessor:
    """Process dataset metadata"""
    
    def __init__(self, dataset_dir: Path):
        """Initialize metadata processor"""
        self.dataset_dir = Path(dataset_dir)
    
    def load_metadata(self, split: str = 'train') -> Dict:
        """Load metadata for a split"""
        metadata_path = self.dataset_dir / split / 'metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def get_sample_ids(self, split: str = 'train') -> List[str]:
        """Get all sample IDs for a split"""
        metadata = self.load_metadata(split)
        return [sample['id'] for sample in metadata['samples']]
    
    def get_sample_label(self, sample_id: str, split: str = 'train') -> int:
        """Get label for a sample"""
        metadata = self.load_metadata(split)
        
        for sample in metadata['samples']:
            if sample['id'] == sample_id:
                return sample['label']
        
        raise ValueError(f"Sample not found: {sample_id}")
    
    def get_all_labels(self, split: str = 'train') -> List[int]:
        """Get all labels for a split"""
        metadata = self.load_metadata(split)
        return [sample['label'] for sample in metadata['samples']]
