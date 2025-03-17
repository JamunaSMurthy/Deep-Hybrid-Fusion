"""
Deep Hybrid Fusion (DHF) Implementation

Complete implementation for multimodal sentiment analysis combining:
- Text Feature Extraction (BERT)
- Image Feature Extraction (HOG + ResNet-101)
- Audio Feature Extraction (LPCC + MFCC)
- Modality Representation (Domain encoders)
- Deep Hybrid Fusion (Transformer-based)
"""

from config import get_config
from Preprocessing import (DatasetDownloader, create_sample_dataset, DataPreprocessor, MetadataProcessor)
from TextFeatureExtraction import TextFeatureExtractor, create_text_feature_extractor
from ImageFeatureExtraction import HybridImageFeatureExtractor, create_image_feature_extractor
from AudioFeatureExtraction import HybridAudioFeatureExtractor, create_audio_feature_extractor
from ModalityRepresentation import ModalityRepresentation
from DeepHybridFusion import DeepHybridFusionModel, create_dhf_model, DHFDataset, create_dhf_dataloader
from train import DHFTrainer, DHFInference

__version__ = "1.0.0"
__author__ = "Deep Hybrid Fusion Team"

__all__ = [
    "get_config",
    "DatasetDownloader",
    "create_sample_dataset",
    "DataPreprocessor",
    "MetadataProcessor",
    "TextFeatureExtractor",
    "create_text_feature_extractor",
    "HybridImageFeatureExtractor",
    "create_image_feature_extractor",
    "HybridAudioFeatureExtractor",
    "create_audio_feature_extractor",
    "ModalityRepresentation",
    "DeepHybridFusionModel",
    "create_dhf_model",
    "DHFDataset",
    "create_dhf_dataloader",
    "DHFTrainer",
    "DHFInference"
]
