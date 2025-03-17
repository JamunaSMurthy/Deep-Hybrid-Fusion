import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Union, List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class TextFeatureExtractor(nn.Module):
    """Extract text features using Enhanced BERT"""
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 max_length: int = 128,
                 output_dim: int = 768,
                 freeze_bert: bool = False,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize text feature extractor
        
        Args:
            model_name: Pretrained BERT model name
            max_length: Maximum token length
            output_dim: Output feature dimension
            freeze_bert: Whether to freeze BERT weights
            device: Device to use (cuda/cpu)
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.output_dim = output_dim
        self.device = device
        
        # Load BERT tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        
        # Get BERT output dimension
        bert_dim = self.bert.config.hidden_size
        
        # Optional projection layer
        if bert_dim != output_dim:
            self.projection = nn.Linear(bert_dim, output_dim)
        else:
            self.projection = None
        
        # Freeze BERT if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.to(device)
    
    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Extract features from text
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Features tensor of shape (B, L, D) or (L, D)
        """
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Use last hidden state
        embeddings = outputs.last_hidden_state  # (B, L, D)
        
        # Project if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # Apply masking
        embeddings = embeddings * attention_mask.unsqueeze(-1)
        
        if return_single:
            return embeddings[0]
        
        return embeddings
    
    def get_sentence_embedding(self, texts: Union[str, List[str]],
                               pooling: str = 'mean') -> torch.Tensor:
        """
        Get sentence-level embeddings
        
        Args:
            texts: Single text or list of texts
            pooling: Pooling method ('mean', 'max', 'cls')
            
        Returns:
            Sentence embeddings of shape (B, D) or (D,)
        """
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
        
        # Get token embeddings
        embeddings = self.forward(texts)  # (B, L, D)
        
        # Pooling
        if pooling == 'mean':
            # Mean pooling with attention mask
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            attention_mask = tokenized['attention_mask'].to(self.device)
            
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = (embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        
        elif pooling == 'max':
            embeddings = embeddings.max(dim=1)[0]
        
        elif pooling == 'cls':
            embeddings = embeddings[:, 0, :]  # [CLS] token
        
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        if return_single:
            return embeddings[0]
        
        return embeddings
    
    def extract_batch(self, texts: List[str], batch_size: int = 32, 
                     pooling: str = 'mean') -> np.ndarray:
        """
        Extract features for a batch of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            pooling: Pooling method
            
        Returns:
            Features array of shape (N, D)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            embeddings = self.get_sentence_embedding(batch_texts, pooling=pooling)
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


class EnhancedBERT(nn.Module):
    """Enhanced BERT with additional layers for sentiment analysis"""
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 max_length: int = 128,
                 output_dim: int = 768,
                 num_classes: int = 2,
                 dropout: float = 0.2,
                 freeze_bert: bool = False,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize Enhanced BERT"""
        super().__init__()
        
        self.feature_extractor = TextFeatureExtractor(
            model_name=model_name,
            max_length=max_length,
            output_dim=output_dim,
            freeze_bert=freeze_bert,
            device=device
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, num_classes)
    
    def forward(self, texts: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            texts: Input texts
            
        Returns:
            Tuple of (embeddings, logits)
        """
        # Get embeddings
        embeddings = self.feature_extractor.get_sentence_embedding(texts, pooling='mean')
        
        # Classification
        hidden = self.dropout(embeddings)
        logits = self.classifier(hidden)
        
        return embeddings, logits


def create_text_feature_extractor(config: Dict) -> TextFeatureExtractor:
    """
    Create text feature extractor from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TextFeatureExtractor instance
    """
    return TextFeatureExtractor(
        model_name=config.get('model_name', 'bert-base-uncased'),
        max_length=config.get('max_length', 128),
        output_dim=config.get('text_dim', 768),
        freeze_bert=config.get('freeze_bert', False)
    )
