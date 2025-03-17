import torch
import torch.nn as nn
from typing import Tuple, Dict, List
import math


class DomainUnchangedEncoder(nn.Module):
    """Domain Unchanged Encoder - extracts common/shared semantic features"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        """
        Initialize Domain Unchanged Encoder
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build encoder network
        layers = []
        prev_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, output_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = output_dim
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to common representation
        
        Args:
            x: Input tensor of shape (B, L, D) or (B, D)
            
        Returns:
            Encoded tensor of shape (B, L, output_dim) or (B, output_dim)
        """
        if len(x.shape) == 3:
            # Sequence input (B, L, D)
            batch_size, seq_len, _ = x.shape
            x_flat = x.view(-1, self.input_dim)
            encoded = self.encoder(x_flat)
            return encoded.view(batch_size, seq_len, self.output_dim)
        else:
            # Single input (B, D)
            return self.encoder(x)


class DomainPreciseEncoder(nn.Module):
    """Domain Precise Encoder - extracts modality-specific/unique features"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        """
        Initialize Domain Precise Encoder
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build encoder network
        layers = []
        prev_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, output_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = output_dim
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to modality-specific representation
        
        Args:
            x: Input tensor of shape (B, L, D) or (B, D)
            
        Returns:
            Encoded tensor of shape (B, L, output_dim) or (B, output_dim)
        """
        if len(x.shape) == 3:
            # Sequence input (B, L, D)
            batch_size, seq_len, _ = x.shape
            x_flat = x.view(-1, self.input_dim)
            encoded = self.encoder(x_flat)
            return encoded.view(batch_size, seq_len, self.output_dim)
        else:
            # Single input (B, D)
            return self.encoder(x)


class ModalityRepresentation(nn.Module):
    """Modality Representation Module - combines common and specific features"""
    
    def __init__(self,
                 text_dim: int = 768,
                 image_dim: int = 2048,
                 audio_dim: int = 40,
                 hidden_dim: int = 256,
                 dropout: float = 0.2):
        """
        Initialize Modality Representation
        
        Args:
            text_dim: Text feature dimension
            image_dim: Image feature dimension
            audio_dim: Audio feature dimension
            hidden_dim: Hidden/output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        
        # Text encoders
        self.text_unchanged = DomainUnchangedEncoder(
            text_dim, hidden_dim, num_layers=2, dropout=dropout
        )
        self.text_precise = DomainPreciseEncoder(
            text_dim, hidden_dim, num_layers=2, dropout=dropout
        )
        
        # Image encoders
        self.image_unchanged = DomainUnchangedEncoder(
            image_dim, hidden_dim, num_layers=2, dropout=dropout
        )
        self.image_precise = DomainPreciseEncoder(
            image_dim, hidden_dim, num_layers=2, dropout=dropout
        )
        
        # Audio encoders
        self.audio_unchanged = DomainUnchangedEncoder(
            audio_dim, hidden_dim, num_layers=2, dropout=dropout
        )
        self.audio_precise = DomainPreciseEncoder(
            audio_dim, hidden_dim, num_layers=2, dropout=dropout
        )
        
        # Normalization layers
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self,
                text_features: torch.Tensor,
                image_features: torch.Tensor,
                audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode multimodal features
        
        Args:
            text_features: Text features of shape (B, L, text_dim) or (B, text_dim)
            image_features: Image features of shape (B, L, image_dim) or (B, image_dim)
            audio_features: Audio features of shape (B, L, audio_dim) or (B, audio_dim)
            
        Returns:
            Tuple of (text_repr, image_repr, audio_repr)
            Each of shape (B, L, hidden_dim) or (B, hidden_dim)
        """
        # Text representation
        text_common = self.text_unchanged(text_features)  # h_t^c
        text_specific = self.text_precise(text_features)  # h_t^p
        text_repr = text_common + text_specific  # h_t = h_t^c + h_t^p
        text_repr = self.norm(text_repr)
        
        # Image representation
        image_common = self.image_unchanged(image_features)  # h_i^c
        image_specific = self.image_precise(image_features)  # h_i^p
        image_repr = image_common + image_specific  # h_i = h_i^c + h_i^p
        image_repr = self.norm(image_repr)
        
        # Audio representation
        audio_common = self.audio_unchanged(audio_features)  # h_a^c
        audio_specific = self.audio_precise(audio_features)  # h_a^p
        audio_repr = audio_common + audio_specific  # h_a = h_a^c + h_a^p
        audio_repr = self.norm(audio_repr)
        
        return text_repr, image_repr, audio_repr
    
    def get_specific_features(self,
                             text_features: torch.Tensor,
                             image_features: torch.Tensor,
                             audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get modality-specific features"""
        text_specific = self.text_precise(text_features)
        image_specific = self.image_precise(image_features)
        audio_specific = self.audio_precise(audio_features)
        
        return text_specific, image_specific, audio_specific
    
    def get_common_features(self,
                           text_features: torch.Tensor,
                           image_features: torch.Tensor,
                           audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get shared/common features"""
        text_common = self.text_unchanged(text_features)
        image_common = self.image_unchanged(image_features)
        audio_common = self.audio_unchanged(audio_features)
        
        return text_common, image_common, audio_common


class ModalityAlignmentLoss(nn.Module):
    """Loss for aligning modality representations"""
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize alignment loss
        
        Args:
            temperature: Temperature for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self,
                text_repr: torch.Tensor,
                image_repr: torch.Tensor,
                audio_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment loss between modalities
        
        Args:
            text_repr: Text representation of shape (B, D)
            image_repr: Image representation of shape (B, D)
            audio_repr: Audio representation of shape (B, D)
            
        Returns:
            Alignment loss
        """
        # Normalize representations
        text_norm = torch.nn.functional.normalize(text_repr, dim=-1)
        image_norm = torch.nn.functional.normalize(image_repr, dim=-1)
        audio_norm = torch.nn.functional.normalize(audio_repr, dim=-1)
        
        # Compute similarity matrices
        similarity_ti = torch.mm(text_norm, image_norm.t()) / self.temperature
        similarity_ta = torch.mm(text_norm, audio_norm.t()) / self.temperature
        similarity_ia = torch.mm(image_norm, audio_norm.t()) / self.temperature
        
        # Target labels (diagonal should be 1)
        batch_size = text_norm.shape[0]
        targets = torch.arange(batch_size, device=text_norm.device)
        
        # Cross-entropy loss
        loss_ti = torch.nn.functional.cross_entropy(similarity_ti, targets)
        loss_ta = torch.nn.functional.cross_entropy(similarity_ta, targets)
        loss_ia = torch.nn.functional.cross_entropy(similarity_ia, targets)
        
        return (loss_ti + loss_ta + loss_ia) / 3.0


class ModalityDiversityLoss(nn.Module):
    """Loss for encouraging modality diversity"""
    
    def __init__(self, lambda_param: float = 0.1):
        """
        Initialize diversity loss
        
        Args:
            lambda_param: Lambda parameter
        """
        super().__init__()
        self.lambda_param = lambda_param
    
    def forward(self,
                text_specific: torch.Tensor,
                image_specific: torch.Tensor,
                audio_specific: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss
        
        Args:
            text_specific: Text-specific features of shape (B, D)
            image_specific: Image-specific features of shape (B, D)
            audio_specific: Audio-specific features of shape (B, D)
            
        Returns:
            Diversity loss
        """
        # Normalize
        text_norm = torch.nn.functional.normalize(text_specific, dim=-1)
        image_norm = torch.nn.functional.normalize(image_specific, dim=-1)
        audio_norm = torch.nn.functional.normalize(audio_specific, dim=-1)
        
        # Compute similarity (should be low for diversity)
        sim_ti = torch.abs(torch.sum(text_norm * image_norm, dim=-1)).mean()
        sim_ta = torch.abs(torch.sum(text_norm * audio_norm, dim=-1)).mean()
        sim_ia = torch.abs(torch.sum(image_norm * audio_norm, dim=-1)).mean()
        
        # Loss: penalize high similarity
        loss = self.lambda_param * (sim_ti + sim_ta + sim_ia)
        
        return loss
