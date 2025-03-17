"""
DHF Inference Script
Complete inference pipeline for making predictions on new multimodal data
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json
from datetime import datetime

from config import get_config
from TextFeatureExtraction import create_text_feature_extractor
from ImageFeatureExtraction import create_image_feature_extractor
from AudioFeatureExtraction import create_audio_feature_extractor
from Preprocessing import DataPreprocessor
from DeepHybridFusion import create_dhf_model
from train import DHFInference


class DHFInferencePipeline:
    """
    Complete inference pipeline for DHF model
    
    Handles:
    - Feature extraction from raw data
    - Feature alignment and preprocessing
    - Model inference
    - Result formatting and confidence calibration
    """
    
    def __init__(self, config_path: str = 'config/config.yaml',
                 model_checkpoint: str = None, device: str = 'cuda'):
        """
        Initialize inference pipeline
        
        Args:
            config_path: Path to configuration file
            model_checkpoint: Path to saved model
            device: Device to use ('cuda' or 'cpu')
        """
        self.config = get_config(config_path)
        self.device = device
        
        # Initialize feature extractors
        print("\nInitializing feature extractors...")
        self.text_extractor = create_text_feature_extractor(
            self.config['text_extraction']
        )
        
        self.image_extractor = create_image_feature_extractor(
            self.config['image_extraction']
        )
        
        self.audio_extractor = create_audio_feature_extractor(
            self.config['audio_extraction']
        )
        
        print("✓ Feature extractors initialized")
        
        # Load model
        print("\nLoading DHF model...")
        if model_checkpoint is None:
            model_checkpoint = self.config['models']['best_model_path']
        
        self.model_checkpoint = model_checkpoint
        self.model = create_dhf_model(self.config['model']).to(device)
        
        # Load weights
        if os.path.exists(model_checkpoint):
            checkpoint = torch.load(model_checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✓ Model loaded from {model_checkpoint}")
        else:
            print(f"⚠ Model checkpoint not found: {model_checkpoint}")
            print("  Using randomly initialized model")
        
        self.model.eval()
        
        # Initialize inference engine
        self.inference = DHFInference(self.model, model_checkpoint, device)
        
        print("✓ Inference pipeline ready")
    
    def extract_features(self, texts: List[str], image_paths: List[str] = None,
                        audio_paths: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract features from raw multimodal data
        
        Args:
            texts: List of text samples
            image_paths: List of image file paths (or None for dummy features)
            audio_paths: List of audio file paths (or None for dummy features)
            
        Returns:
            Dict with keys 'text', 'image', 'audio' containing feature arrays
        """
        n_samples = len(texts)
        
        # Extract text features
        print(f"\nExtracting text features ({n_samples} samples)...")
        text_features = self.text_extractor.extract_batch(texts)
        print(f"✓ Text features: {text_features.shape}")
        
        # Extract image features
        if image_paths and len(image_paths) > 0:
            print(f"Extracting image features ({len(image_paths)} images)...")
            image_features = self.image_extractor.extract_batch(image_paths)
            print(f"✓ Image features: {image_features.shape}")
        else:
            # Use dummy features if paths not provided
            image_dim = self.config['model']['image_dim']
            image_features = np.random.randn(n_samples, image_dim).astype(np.float32)
            print(f"⚠ Using dummy image features: {image_features.shape}")
        
        # Extract audio features
        if audio_paths and len(audio_paths) > 0:
            print(f"Extracting audio features ({len(audio_paths)} audios)...")
            audio_features = self.audio_extractor.extract_batch(audio_paths)
            print(f"✓ Audio features: {audio_features.shape}")
        else:
            # Use dummy features if paths not provided
            audio_dim = self.config['model']['audio_dim']
            audio_features = np.random.randn(n_samples, audio_dim).astype(np.float32)
            print(f"⚠ Using dummy audio features: {audio_features.shape}")
        
        return {
            'text': text_features,
            'image': image_features,
            'audio': audio_features,
        }
    
    def preprocess_features(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess extracted features
        
        Args:
            features: Dict with extracted features
            
        Returns:
            Dict with preprocessed features
        """
        print("\nPreprocessing features...")
        
        preprocessor = DataPreprocessor()
        
        # Normalize each modality
        preprocessed = {}
        for modality, feature_array in features.items():
            normalized = preprocessor.normalize_features(
                feature_array, method='standard'
            )
            preprocessed[modality] = normalized
            print(f"✓ {modality.capitalize()} normalized: {normalized.shape}")
        
        # Align modalities to same temporal length
        if preprocessed['text'].ndim == 3:  # (B, T, D)
            target_length = min(
                preprocessed['text'].shape[1],
                preprocessed['image'].shape[1] if preprocessed['image'].ndim == 3 else 1,
                preprocessed['audio'].shape[1] if preprocessed['audio'].ndim == 3 else 1,
            )
            
            text_aligned = preprocessor.pad_or_truncate(
                preprocessed['text'], target_length
            )
            image_aligned = preprocessor.pad_or_truncate(
                preprocessed['image'], target_length
            )
            audio_aligned = preprocessor.pad_or_truncate(
                preprocessed['audio'], target_length
            )
            
            preprocessed = {
                'text': text_aligned,
                'image': image_aligned,
                'audio': audio_aligned,
            }
            print(f"✓ Modalities aligned to length {target_length}")
        
        return preprocessed
    
    def predict(self, texts: List[str], image_paths: List[str] = None,
               audio_paths: List[str] = None, return_embeddings: bool = False) -> Dict:
        """
        Complete inference pipeline: extract → preprocess → predict
        
        Args:
            texts: List of text samples
            image_paths: List of image paths
            audio_paths: List of audio paths
            return_embeddings: Whether to return fused embeddings
            
        Returns:
            Dict with predictions and confidence scores
        """
        print("\n" + "="*70)
        print(" "*15 + "DHF INFERENCE PIPELINE")
        print("="*70)
        
        # Extract features
        features = self.extract_features(texts, image_paths, audio_paths)
        
        # Preprocess
        preprocessed = self.preprocess_features(features)
        
        # Convert to tensors
        print("\nPreparing tensors...")
        text_tensor = torch.from_numpy(preprocessed['text']).float().to(self.device)
        image_tensor = torch.from_numpy(preprocessed['image']).float().to(self.device)
        audio_tensor = torch.from_numpy(preprocessed['audio']).float().to(self.device)
        
        # Make predictions
        print("Making predictions...")
        with torch.no_grad():
            if isinstance(self.inference, DHFInference):
                predictions, probabilities = self.inference.predict_batch(
                    preprocessed['text'], preprocessed['image'], preprocessed['audio']
                )
            else:
                logits, embeddings = self.model(text_tensor, image_tensor, audio_tensor)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
        
        print("✓ Predictions generated")
        
        # Format results
        results = self._format_results(
            texts, predictions, probabilities,
            return_embeddings=return_embeddings
        )
        
        return results
    
    def _format_results(self, texts: List[str], predictions: np.ndarray,
                       probabilities: np.ndarray, return_embeddings: bool = False) -> Dict:
        """Format prediction results"""
        
        class_names = ['Negative', 'Positive']
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_checkpoint': self.model_checkpoint,
            'num_samples': len(texts),
            'predictions': [],
        }
        
        for i, (text, pred, probs) in enumerate(zip(texts, predictions, probabilities)):
            sample_result = {
                'sample_id': i,
                'text': text,
                'predicted_class': class_names[pred],
                'predicted_index': int(pred),
                'confidence': float(probs[pred]),
                'probabilities': {
                    class_names[j]: float(probs[j])
                    for j in range(len(class_names))
                }
            }
            results['predictions'].append(sample_result)
        
        # Summary statistics
        correct_pred_count = np.sum(probabilities[np.arange(len(predictions)), predictions] > 0.5)
        results['summary'] = {
            'total_samples': len(predictions),
            'high_confidence_predictions': int(correct_pred_count),
            'mean_confidence': float(np.mean(probabilities.max(axis=1))),
            'std_confidence': float(np.std(probabilities.max(axis=1))),
        }
        
        return results
    
    def save_results(self, results: Dict, output_path: str = 'outputs/predictions.json'):
        """Save prediction results to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    
    def print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "="*70)
        print(" "*20 + "INFERENCE RESULTS")
        print("="*70)
        
        print(f"\nTotal Samples: {results['summary']['total_samples']}")
        print(f"Mean Confidence: {results['summary']['mean_confidence']:.4f}")
        print(f"Std Confidence: {results['summary']['std_confidence']:.4f}")
        
        print("\n" + "-"*70)
        print("Sample Predictions:")
        print("-"*70)
        
        for pred in results['predictions'][:5]:  # Show first 5
            print(f"\nSample {pred['sample_id']}:")
            print(f"  Text: {pred['text'][:60]}...")
            print(f"  Prediction: {pred['predicted_class']}")
            print(f"  Confidence: {pred['confidence']:.4f}")
            if len(results['predictions']) > 5:
                print("  ...")
                break
        
        if len(results['predictions']) > 5:
            print(f"\n(Showing 5/{len(results['predictions'])} predictions)")
        
        print("\n" + "="*70)


def main():
    """Example inference usage"""
    
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "Deep Hybrid Fusion - Inference Pipeline".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    # Initialize pipeline
    try:
        pipeline = DHFInferencePipeline(device='cuda')
    except Exception as e:
        print(f"\nError: {e}")
        print("Using CPU instead...")
        pipeline = DHFInferencePipeline(device='cpu')
    
    # Example texts
    test_texts = [
        "This movie is absolutely amazing! I loved every minute of it.",
        "I didn't like this film at all. It was boring and predictable.",
        "This is an excellent production with great acting.",
        "The worst movie I've ever watched.",
        "Not bad, but could have been better.",
    ]
    
    print(f"\n\nExample 1: Predict on test samples")
    print("-"*70)
    results = pipeline.predict(test_texts)
    pipeline.print_results(results)
    
    # Save results
    print(f"\nExample 2: Save results to file")
    print("-"*70)
    pipeline.save_results(results, 'outputs/example_predictions.json')
    
    # Batch inference
    print(f"\n\nExample 3: Batch inference with multiple samples")
    print("-"*70)
    batch_texts = test_texts * 2  # Repeat for larger batch
    results = pipeline.predict(batch_texts)
    print(f"✓ Processed {len(results['predictions'])} samples")
    print(f"  Mean confidence: {results['summary']['mean_confidence']:.4f}")
    
    print(f"\n✓ Inference pipeline demonstration complete!")


if __name__ == '__main__':
    main()
