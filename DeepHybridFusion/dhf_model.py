import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize MultiHead Attention
        
        Args:
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query tensor of shape (B, L, hidden_dim)
            key: Key tensor of shape (B, L, hidden_dim)
            value: Value tensor of shape (B, L, hidden_dim)
            mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.query(query)  # (B, L, hidden_dim)
        K = self.key(key)      # (B, L, hidden_dim)
        V = self.value(value)  # (B, L, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, num_heads, L, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, num_heads, L, L)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)  # (B, num_heads, L, head_dim)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_dim)  # (B, L, hidden_dim)
        
        # Final linear transformation
        output = self.fc_out(context)
        
        return output, attention.mean(dim=1)  # Average attention across heads


class FeedForwardNetwork(nn.Module):
    """Feed Forward Network"""
    
    def __init__(self,
                 hidden_dim: int,
                 ff_dim: int,
                 dropout: float = 0.1):
        """Initialize FFN"""
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Transformer Block with attention and FFN"""
    
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 ff_dim: int = 2048,
                 dropout: float = 0.1):
        """Initialize Transformer Block"""
        super().__init__()
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, ff_dim, dropout)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections"""
        # Self-attention
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusion"""
    
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """Initialize cross-modal attention"""
        super().__init__()
        
        self.text_to_image = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.text_to_audio = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.image_to_text = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.image_to_audio = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.audio_to_text = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.audio_to_image = MultiHeadAttention(hidden_dim, num_heads, dropout)
    
    def forward(self,
                text_repr: torch.Tensor,
                image_repr: torch.Tensor,
                audio_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cross-modal attention fusion
        
        Args:
            text_repr: Text representation (B, L, D) or (B, D)
            image_repr: Image representation (B, L, D) or (B, D)
            audio_repr: Audio representation (B, L, D) or (B, D)
            
        Returns:
            Fused representations
        """
        # Handle both sequence and non-sequence inputs
        is_sequence = len(text_repr.shape) == 3
        
        if not is_sequence:
            text_repr = text_repr.unsqueeze(1)
            image_repr = image_repr.unsqueeze(1)
            audio_repr = audio_repr.unsqueeze(1)
        
        # Cross-modal attention
        text_attended, _ = self.text_to_image(text_repr, image_repr, image_repr)
        text_attended, _ = self.text_to_audio(text_attended, audio_repr, audio_repr)
        
        image_attended, _ = self.image_to_text(image_repr, text_repr, text_repr)
        image_attended, _ = self.image_to_audio(image_attended, audio_repr, audio_repr)
        
        audio_attended, _ = self.audio_to_text(audio_repr, text_repr, text_repr)
        audio_attended, _ = self.audio_to_image(audio_attended, image_repr, image_repr)
        
        if not is_sequence:
            text_attended = text_attended.squeeze(1)
            image_attended = image_attended.squeeze(1)
            audio_attended = audio_attended.squeeze(1)
        
        return text_attended, image_attended, audio_attended


class FusionLayer(nn.Module):
    """Fusion layer for combining multimodal representations"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_modalities: int = 3):
        """Initialize fusion layer"""
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_modalities = num_modalities
        
        # Fusion weights for each modality
        self.fusion_weights = nn.Parameter(torch.ones(num_modalities))
        
        # Projection layer
        self.projection = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, representations: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multimodal representations
        
        Args:
            representations: Tuple of (text, image, audio) representations
            
        Returns:
            Fused representation
        """
        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Weighted fusion
        text_repr, image_repr, audio_repr = representations
        
        fused = (weights[0] * text_repr +
                weights[1] * image_repr +
                weights[2] * audio_repr)
        
        # Project and normalize
        fused = self.projection(fused)
        fused = self.norm(fused)
        
        return fused


class DeepHybridFusionModel(nn.Module):
    """Deep Hybrid Fusion Model for multimodal sentiment analysis"""
    
    def __init__(self,
                 text_dim: int = 768,
                 image_dim: int = 2048,
                 audio_dim: int = 40,
                 hidden_dim: int = 256,
                 num_fusion_layers: int = 3,
                 num_attention_heads: int = 8,
                 output_dim: int = 2,
                 dropout: float = 0.1):
        """
        Initialize Deep Hybrid Fusion Model
        
        Args:
            text_dim: Text feature dimension
            image_dim: Image feature dimension
            audio_dim: Audio feature dimension
            hidden_dim: Hidden dimension
            num_fusion_layers: Number of fusion layers
            num_attention_heads: Number of attention heads
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection layers
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            hidden_dim, num_attention_heads, dropout
        )
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            FusionLayer(hidden_dim, hidden_dim // (2 ** i))
            for i in range(num_fusion_layers)
        ])
        
        # Final fusion dimension
        final_dim = hidden_dim // (2 ** (num_fusion_layers - 1))
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self,
                text_features: torch.Tensor,
                image_features: torch.Tensor,
                audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            text_features: Text features (B, text_dim)
            image_features: Image features (B, image_dim)
            audio_features: Audio features (B, audio_dim)
            
        Returns:
            Tuple of (logits, fused_repr)
        """
        # Project inputs to common dimension
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        audio_proj = self.audio_proj(audio_features)
        
        # Normalize
        text_proj = self.norm(text_proj)
        image_proj = self.norm(image_proj)
        audio_proj = self.norm(audio_proj)
        
        # Cross-modal attention
        text_attn, image_attn, audio_attn = self.cross_modal_attention(
            text_proj, image_proj, audio_proj
        )
        
        # Residual connections
        text_repr = text_proj + text_attn
        image_repr = image_proj + image_attn
        audio_repr = audio_proj + audio_attn
        
        # Fusion layers
        fused = (text_repr, image_repr, audio_repr)
        
        for fusion_layer in self.fusion_layers:
            fused_output = fusion_layer(fused)
            fused = (fused_output, fused_output, fused_output)
        
        # Get final fused representation
        if isinstance(fused, tuple):
            final_repr = fused[0]
        else:
            final_repr = fused
        
        # Decode for classification
        logits = self.decoder(final_repr)
        
        return logits, final_repr
    
    def get_attention_weights(self,
                             text_features: torch.Tensor,
                             image_features: torch.Tensor,
                             audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get attention weights for visualization"""
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        audio_proj = self.audio_proj(audio_features)
        
        text_proj = self.norm(text_proj)
        image_proj = self.norm(image_proj)
        audio_proj = self.norm(audio_proj)
        
        # We would need to modify MultiHeadAttention to return weights
        # For now, this is a placeholder
        
        return {}


def create_dhf_model(config: Dict) -> DeepHybridFusionModel:
    """Create DHF model from config"""
    return DeepHybridFusionModel(
        text_dim=config.get('text_dim', 768),
        image_dim=config.get('image_dim', 2048),
        audio_dim=config.get('audio_dim', 40),
        hidden_dim=config.get('fusion_dim', 256),
        num_fusion_layers=config.get('num_fusion_layers', 3),
        num_attention_heads=config.get('num_attention_heads', 8),
        output_dim=config.get('output_dim', 2),
        dropout=config.get('encoder_dropout', 0.1)
    )
