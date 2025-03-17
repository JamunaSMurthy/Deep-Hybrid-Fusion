#!/usr/bin/env python3
"""
Complete DHF Training/Inference Example
Demonstrates the full pipeline for text, image, and audio feature extraction
and multimodal fusion using Deep Hybrid Fusion architecture.
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from Preprocessing import (
    DatasetDownloader, create_sample_dataset,
    DataPreprocessor, MetadataProcessor
)
from TextFeatureExtraction import create_text_feature_extractor
from ImageFeatureExtraction import create_image_feature_extractor
from AudioFeatureExtraction import create_audio_feature_extractor
from ModalityRepresentation import ModalityRepresentation
from DeepHybridFusion import create_dhf_model, create_dhf_dataloader
from train import DHFTrainer

def setup_data():
    """Setup and download dataset"""
    print("=" * 70)
    print("STEP 1: Setting up dataset...")
    print("=" * 70)
    
    # Create sample dataset for demonstration
    dataset_dir = create_sample_dataset("./data", "MOSI")
    
    return dataset_dir


def extract_features(dataset_dir):
    """Extract features from all modalities"""
    print("\n" + "=" * 70)
    print("STEP 2: Extracting multimodal features...")
    print("=" * 70)
    
    # Load configuration
    config = get_config()
    
    # Initialize feature extractors
    print("\n[2.1] Initializing feature extractors...")
    text_extractor = create_text_feature_extractor(config['text_extraction'])
    image_extractor = create_image_feature_extractor(config['image_extraction'])
    audio_extractor = create_audio_feature_extractor(config['audio_extraction'])
    
    # Extract text features (example texts)
    print("\n[2.2] Extracting text features...")
    sample_texts = [
        "This movie is absolutely amazing!",
        "I really didn't like this film at all.",
        "It was an okay movie, nothing special.",
        "The best movie I've ever seen!",
        "Terrible, waste of time!"
    ]
    
    text_features = text_extractor.extract_batch(sample_texts)
    print(f"Text features shape: {text_features.shape}")  # (5, 768)
    
    # Create dummy image and audio features for demonstration
    print("\n[2.3] Creating dummy image features...")
    image_features = np.random.randn(5, 2048).astype(np.float32)
    print(f"Image features shape: {image_features.shape}")  # (5, 2048)
    
    print("\n[2.4] Creating dummy audio features...")
    audio_features = np.random.randn(5, 40).astype(np.float32)
    print(f"Audio features shape: {audio_features.shape}")  # (5, 40)
    
    # Labels
    labels = np.array([1, 0, 0, 1, 0], dtype=np.int64)
    print(f"Labels: {labels}")
    
    # Normalize features
    print("\n[2.5] Normalizing features...")
    text_features = DataPreprocessor.normalize_features(text_features, 'standard')
    image_features = DataPreprocessor.normalize_features(image_features, 'standard')
    audio_features = DataPreprocessor.normalize_features(audio_features, 'standard')
    
    print("✓ Feature extraction completed!")
    
    return text_features, image_features, audio_features, labels


def prepare_data(text_features, image_features, audio_features, labels):
    """Prepare data for training"""
    print("\n" + "=" * 70)
    print("STEP 3: Preparing data for training...")
    print("=" * 70)
    
    # Split data
    n_samples = len(labels)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Create datasets
    config = get_config()
    
    train_loader = create_dhf_dataloader(
        text_features[train_idx],
        image_features[train_idx],
        audio_features[train_idx],
        labels[train_idx],
        batch_size=config['dataset']['batch_size'],
        shuffle=True
    )
    
    val_loader = create_dhf_dataloader(
        text_features[val_idx],
        image_features[val_idx],
        audio_features[val_idx],
        labels[val_idx],
        batch_size=config['dataset']['batch_size'],
        shuffle=False
    )
    
    test_loader = create_dhf_dataloader(
        text_features[test_idx],
        image_features[test_idx],
        audio_features[test_idx],
        labels[test_idx],
        batch_size=config['dataset']['batch_size'],
        shuffle=False
    )
    
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print("✓ Data preparation completed!")
    
    return train_loader, val_loader, test_loader


def create_model(config):
    """Create DHF model"""
    print("\n" + "=" * 70)
    print("STEP 4: Creating Deep Hybrid Fusion model...")
    print("=" * 70)
    
    # Create modality representation module
    modality_repr = ModalityRepresentation(
        text_dim=config['model']['text_dim'],
        image_dim=config['model']['image_dim'],
        audio_dim=config['model']['audio_dim'],
        hidden_dim=config['model']['encoder_hidden_dim'],
        dropout=config['model']['encoder_dropout']
    )
    
    print(f"Modality Representation created with hidden_dim={config['model']['encoder_hidden_dim']}")
    
    # Create DHF model
    dhf_model = create_dhf_model(config['model'])
    
    print(f"Deep Hybrid Fusion model created:")
    print(f"  - Text dimension: {config['model']['text_dim']}")
    print(f"  - Image dimension: {config['model']['image_dim']}")
    print(f"  - Audio dimension: {config['model']['audio_dim']}")
    print(f"  - Fusion dimension: {config['model']['fusion_dim']}")
    print(f"  - Output dimension: {config['model']['output_dim']}")
    
    # Count parameters
    param_count = sum(p.numel() for p in dhf_model.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {param_count:,}")
    
    print("✓ Model creation completed!")
    
    return modality_repr, dhf_model


def train_model(model, train_loader, val_loader, config):
    """Train the DHF model"""
    print("\n" + "=" * 70)
    print("STEP 5: Training Deep Hybrid Fusion model...")
    print("=" * 70)
    
    device = config['training']['device']
    
    # Create trainer
    trainer = DHFTrainer(model, config['training'], device)
    
    # Train
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=config['training']['epochs'],
        save_dir='./models'
    )
    
    return history


def main():
    """Main function"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "Deep Hybrid Fusion (DHF) - Complete Example" + " " * 15 + "║")
    print("║" + " " * 20 + "Multimodal Sentiment Analysis" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        # 1. Setup data
        dataset_dir = setup_data()
        
        # 2. Extract features
        text_features, image_features, audio_features, labels = extract_features(dataset_dir)
        
        # 3. Prepare data
        train_loader, val_loader, test_loader = prepare_data(
            text_features, image_features, audio_features, labels
        )
        
        # 4. Create model
        config = get_config()
        modality_repr, dhf_model = create_model(config['model'])
        
        # 5. Train model
        history = train_model(dhf_model, train_loader, val_loader, config['training'])
        
        print("\n" + "=" * 70)
        print("STEP 6: Evaluation on test set...")
        print("=" * 70)
        
        # Evaluate
        dhf_model.eval()
        device = config['training']['device']
        dhf_model = dhf_model.to(device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for text, image, audio, labels in test_loader:
                text = text.to(device)
                image = image.to(device)
                audio = audio.to(device)
                labels = labels.to(device)
                
                logits, _ = dhf_model(text, image, audio)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = correct / total if total > 0 else 0.0
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        
        print("\n" + "=" * 70)
        print("✓ Complete pipeline executed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
