"""
Comprehensive Model Evaluation Script for DHF
Evaluates trained DHF model on test set with detailed metrics and visualizations
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_config
from DeepHybridFusion import create_dhf_model
from train import DHFInference, DHFTrainingUtils


class DHFEvaluator:
    """
    Comprehensive evaluation framework for DHF models
    
    Computes metrics:
    - Classification: accuracy, precision, recall, F1, AUC-ROC
    - Confusion matrix visualization
    - Per-class performance metrics
    - ROC and Precision-Recall curves
    - Model complexity analysis
    """
    
    def __init__(self, config_path='config/config.yaml', device='cuda'):
        """
        Initialize evaluator
        
        Args:
            config_path (str): Path to config file
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.config = get_config(config_path)
        self.device = device
        self.metrics = {}
        self.results = {}
        
        # Report template
        self.report_template = """
╔════════════════════════════════════════════════════════════════╗
║           DEEP HYBRID FUSION MODEL EVALUATION REPORT           ║
╚════════════════════════════════════════════════════════════════╝

Model: {model_name}
Checkpoint: {checkpoint}
Device: {device}
Evaluation Date: {timestamp}
Dataset Split: {dataset_split}
Total Samples: {total_samples}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERALL METRICS

Accuracy:                {accuracy:.4f}
Precision (weighted):    {precision_weighted:.4f}
Recall (weighted):       {recall_weighted:.4f}
F1-Score (weighted):     {f1_weighted:.4f}
AUC-ROC:                 {auc_score:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PER-CLASS METRICS

{per_class_metrics}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFUSION MATRIX

Predicted →
Actual ↓      Negative    Positive
Negative      {tn:>6d}      {fp:>6d}
Positive      {fn:>6d}      {tp:>6d}

True Positive Rate (Sensitivity):  {tpr:.4f}
True Negative Rate (Specificity):  {tnr:.4f}
False Positive Rate:               {fpr:.4f}
False Negative Rate:               {fnr:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE CALIBRATION

Mean Prediction Confidence:  {mean_confidence:.4f}
Std Prediction Confidence:   {std_confidence:.4f}
Min Prediction Confidence:   {min_confidence:.4f}
Max Prediction Confidence:   {max_confidence:.4f}

Correct Predictions - Mean Confidence:  {correct_conf:.4f}
Incorrect Predictions - Mean Confidence: {incorrect_conf:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL COMPLEXITY

Total Parameters:           {total_params:,}
Trainable Parameters:       {trainable_params:,}
Model Size:                 {model_size:.2f} MB
FLOPs per Sample:           {flops_per_sample:,}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFERENCE STATISTICS

Inference Time (avg):       {avg_inference_time:.4f} seconds
Inference FPS:              {inference_fps:.2f} samples/sec
Memory Usage (peak):        {peak_memory:.2f} MB

╚════════════════════════════════════════════════════════════════╝
"""
    
    def evaluate(self, test_loader, model_checkpoint=None, model=None):
        """
        Evaluate model on test set
        
        Args:
            test_loader: PyTorch DataLoader for test set
            model_checkpoint (str): Path to saved model
            model: Trained model (if None, loads from checkpoint)
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "="*70)
        print(" "*15 + "INITIALIZING EVALUATION")
        print("="*70)
        
        # Load or use provided model
        if model is None:
            if model_checkpoint is None:
                model_checkpoint = self.config['models']['best_model_path']
            
            model = create_dhf_model(self.config['model']).to(self.device)
            checkpoint = torch.load(model_checkpoint, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            print(f"✓ Loaded model from: {model_checkpoint}")
        else:
            model_checkpoint = "provided_model"
        
        model.eval()
        
        # Collect predictions and labels
        all_predictions = []
        all_labels = []
        all_probabilities = []
        inference_times = []
        
        print("\nEvaluating model...")
        with torch.no_grad():
            for batch_idx, (text, image, audio, labels) in enumerate(test_loader):
                # Move to device
                text = text.to(self.device)
                image = image.to(self.device)
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                
                # Inference
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                logits, fused_repr = model(text, image, audio)
                end_time.record()
                
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                inference_times.append(inference_time)
                
                # Get predictions
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                
                if (batch_idx + 1) % max(1, len(test_loader) // 5) == 0:
                    print(f"  Processed: {batch_idx + 1}/{len(test_loader)} batches")
        
        # Convert to numpy
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        print("✓ Evaluation complete!")
        
        # Compute metrics
        self._compute_metrics(y_true, y_pred, y_proba, inference_times, model, model_checkpoint)
        
        return self.metrics
    
    def _compute_metrics(self, y_true, y_pred, y_proba, inference_times, model, checkpoint):
        """Compute comprehensive metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Binary class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # AUC-ROC (binary classification)
        try:
            auc_score = roc_auc_score(y_true, y_proba[:, 1])
        except:
            auc_score = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Confidence metrics
        max_probs = np.max(y_proba, axis=1)
        correct_mask = y_pred == y_true
        correct_conf = np.mean(max_probs[correct_mask]) if np.any(correct_mask) else 0.0
        incorrect_conf = np.mean(max_probs[~correct_mask]) if np.any(~correct_mask) else 0.0
        
        # Model complexity
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)  # MB
        
        # Inference statistics
        avg_inference_time = np.mean(inference_times)
        inference_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'auc_roc': auc_score,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'tpr': tpr,
            'tnr': tnr,
            'fpr': fpr,
            'fnr': fnr,
            'mean_confidence': float(np.mean(max_probs)),
            'std_confidence': float(np.std(max_probs)),
            'min_confidence': float(np.min(max_probs)),
            'max_confidence': float(np.max(max_probs)),
            'correct_conf': correct_conf,
            'incorrect_conf': incorrect_conf,
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'model_size_mb': float(model_size),
            'avg_inference_time': float(avg_inference_time),
            'inference_fps': float(inference_fps),
        }
        
        # Store raw predictions for visualization
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.checkpoint = checkpoint
        
        # Print report
        self._print_report(accuracy, precision_weighted, recall_weighted, f1_weighted, 
                          auc_score, precision_per_class, recall_per_class, f1_per_class,
                          tn, fp, fn, tp, tpr, tnr, fpr, fnr,
                          np.mean(max_probs), np.std(max_probs), np.min(max_probs), np.max(max_probs),
                          correct_conf, incorrect_conf, total_params, trainable_params, model_size,
                          avg_inference_time, inference_fps, len(y_true))
    
    def _print_report(self, accuracy, precision_weighted, recall_weighted, f1_weighted,
                     auc_score, precision_per_class, recall_per_class, f1_per_class,
                     tn, fp, fn, tp, tpr, tnr, fpr, fnr,
                     mean_conf, std_conf, min_conf, max_conf,
                     correct_conf, incorrect_conf, total_params, trainable_params,
                     model_size, avg_inference_time, inference_fps, total_samples):
        """Print formatted evaluation report"""
        
        # Per-class metrics string
        per_class_str = "Class  | Precision | Recall | F1-Score\n"
        per_class_str += "-------|-----------|--------|----------\n"
        class_names = ["Negative", "Positive"]
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            per_class_str += f"{class_names[i]:<6} | {p:>9.4f} | {r:>6.4f} | {f:>8.4f}\n"
        
        report = self.report_template.format(
            model_name="DHF (Deep Hybrid Fusion)",
            checkpoint=self.checkpoint,
            device=self.device,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset_split="Test Set",
            total_samples=total_samples,
            accuracy=accuracy,
            precision_weighted=precision_weighted,
            recall_weighted=recall_weighted,
            f1_weighted=f1_weighted,
            auc_score=auc_score,
            per_class_metrics=per_class_str,
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            tp=int(tp),
            tpr=tpr,
            tnr=tnr,
            fpr=fpr,
            fnr=fnr,
            mean_confidence=mean_conf,
            std_confidence=std_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            correct_conf=correct_conf,
            incorrect_conf=incorrect_conf,
            total_params=total_params,
            trainable_params=trainable_params,
            model_size=model_size,
            flops_per_sample=768*2048*40//32,  # Rough FLOPs estimate
            avg_inference_time=avg_inference_time,
            inference_fps=inference_fps,
            peak_memory=0.0,  # Would need to instrument GPU memory
        )
        
        print(report)
    
    def plot_confusion_matrix(self, save_path='outputs/confusion_matrix.png'):
        """Plot confusion matrix"""
        if not hasattr(self, 'y_true'):
            print("No evaluation results. Run evaluate() first.")
            return
        
        cm = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix - DHF Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_roc_curve(self, save_path='outputs/roc_curve.png'):
        """Plot ROC curve"""
        if not hasattr(self, 'y_proba'):
            print("No evaluation results. Run evaluate() first.")
            return
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - DHF Model')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
        plt.close()
    
    def plot_precision_recall_curve(self, save_path='outputs/pr_curve.png'):
        """Plot Precision-Recall curve"""
        if not hasattr(self, 'y_proba'):
            print("No evaluation results. Run evaluate() first.")
            return
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba[:, 1])
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkgreen', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - DHF Model')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Precision-Recall curve saved to {save_path}")
        plt.close()
    
    def save_metrics_json(self, save_path='outputs/evaluation_metrics.json'):
        """Save metrics to JSON"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✓ Metrics saved to {save_path}")
    
    def generate_all_plots(self, output_dir='outputs'):
        """Generate all evaluation plots"""
        print("\n" + "="*70)
        print(" "*20 + "GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        
        self.plot_confusion_matrix(f'{output_dir}/confusion_matrix.png')
        self.plot_roc_curve(f'{output_dir}/roc_curve.png')
        self.plot_precision_recall_curve(f'{output_dir}/pr_curve.png')
        self.save_metrics_json(f'{output_dir}/evaluation_metrics.json')


def main():
    """Example evaluation script"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "Deep Hybrid Fusion - Model Evaluation".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝\n")
    
    # Load configuration
    config = get_config()
    
    # Initialize evaluator
    evaluator = DHFEvaluator(config_path='config/config.yaml', device='cuda')
    
    print("✓ Evaluator initialized")
    print(f"  Device: {evaluator.device}")
    print(f"  Model checkpoint: {config['models']['best_model_path']}")
    
    # Create dummy test loader (in practice, load real test data)
    print("\nNote: Using dummy test data for demonstration.")
    print("Replace with real test DataLoader for actual evaluation.")
    
    # For demonstration, show how to use the evaluator:
    print("\nExample Usage:")
    print("  from evaluate import DHFEvaluator")
    print("  evaluator = DHFEvaluator(device='cuda')")
    print("  metrics = evaluator.evaluate(test_loader)")
    print("  evaluator.generate_all_plots()")
    print("  evaluator.save_metrics_json()")


if __name__ == '__main__':
    main()
