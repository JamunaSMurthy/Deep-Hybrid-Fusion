import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
import cv2
import numpy as np
from skimage.feature import hog
from pathlib import Path
from typing import Union, List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class HOGFeatureExtractor:
    """Extract HOG (Histogram of Oriented Gradients) features from images"""
    
    def __init__(self,
                 orientations: int = 9,
                 pixels_per_cell: int = 8,
                 cells_per_block: int = 2):
        """
        Initialize HOG extractor
        
        Args:
            orientations: Number of orientation bins
            pixels_per_cell: Pixels per cell
            cells_per_block: Cells per block
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features
        
        Args:
            image: Input image (H, W, C) or (H, W)
            
        Returns:
            HOG feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        
        # Extract HOG features
        features = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
            cells_per_block=(self.cells_per_block, self.cells_per_block),
            block_norm='L2-Hys',
            visualize=False
        )
        
        return features
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract HOG features for batch of images"""
        features = []
        for img in images:
            feat = self.extract(img)
            features.append(feat)
        
        return np.array(features)


class ResNetFeatureExtractor(nn.Module):
    """Extract features using ResNet/MultiNet-101"""
    
    def __init__(self,
                 model_name: str = 'resnet101',
                 pretrained: bool = True,
                 output_dim: int = 2048,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize ResNet feature extractor
        
        Args:
            model_name: ResNet model name (resnet50, resnet101, etc.)
            pretrained: Use pretrained weights
            output_dim: Output feature dimension
            device: Device to use
        """
        super().__init__()
        
        self.device = device
        self.output_dim = output_dim
        
        # Load pretrained ResNet
        if model_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            self.resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove classification layer
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Freeze backbone
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.to(device)
    
    def forward(self, images: Union[np.ndarray, List[np.ndarray], torch.Tensor]) -> torch.Tensor:
        """
        Extract features from images
        
        Args:
            images: Single image or batch of images
            
        Returns:
            Feature tensor of shape (B, 2048) or (2048,)
        """
        if isinstance(images, list):
            # Batch of numpy arrays
            processed = []
            for img in images:
                processed.append(self.preprocess(self._to_pil(img)))
            
            tensor = torch.stack(processed).to(self.device)
            return_single = len(images) == 1
        
        elif isinstance(images, np.ndarray):
            # Single numpy array
            tensor = self.preprocess(self._to_pil(images)).unsqueeze(0).to(self.device)
            return_single = True
        
        elif isinstance(images, torch.Tensor):
            # Already tensor
            tensor = images.to(self.device)
            return_single = False
        
        else:
            raise TypeError(f"Unsupported type: {type(images)}")
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(tensor)
            features = features.view(features.size(0), -1)  # Flatten
        
        if return_single:
            return features[0]
        
        return features
    
    @staticmethod
    def _to_pil(image: np.ndarray):
        """Convert numpy array to PIL Image"""
        from PIL import Image
        
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(image)


class HybridImageFeatureExtractor:
    """Extract features using both HOG and ResNet"""
    
    def __init__(self,
                 use_hog: bool = True,
                 use_resnet: bool = True,
                 resnet_model: str = 'resnet101',
                 hog_orientations: int = 9,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize hybrid image feature extractor
        
        Args:
            use_hog: Whether to use HOG features
            use_resnet: Whether to use ResNet features
            resnet_model: ResNet model type
            hog_orientations: HOG orientations
            device: Device to use
        """
        self.use_hog = use_hog
        self.use_resnet = use_resnet
        
        if use_hog:
            self.hog_extractor = HOGFeatureExtractor(orientations=hog_orientations)
        else:
            self.hog_extractor = None
        
        if use_resnet:
            self.resnet_extractor = ResNetFeatureExtractor(
                model_name=resnet_model,
                device=device
            )
        else:
            self.resnet_extractor = None
        
        self.device = device
    
    def extract(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        Extract hybrid features from image
        
        Args:
            image: Image array or path
            
        Returns:
            Concatenated feature vector
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise FileNotFoundError(f"Image not found: {image}")
        
        features = []
        
        # HOG features
        if self.use_hog:
            hog_feat = self.hog_extractor.extract(image)
            features.append(hog_feat)
        
        # ResNet features
        if self.use_resnet:
            resnet_feat = self.resnet_extractor(image).cpu().numpy()
            features.append(resnet_feat)
        
        # Concatenate
        if features:
            return np.concatenate(features)
        else:
            raise ValueError("At least one feature extractor must be enabled")
    
    def extract_batch(self, images: List[Union[np.ndarray, str]], 
                     batch_size: int = 32) -> np.ndarray:
        """
        Extract features for batch of images
        
        Args:
            images: List of image arrays or paths
            batch_size: Batch size for ResNet processing
            
        Returns:
            Features array of shape (N, D)
        """
        all_features = []
        
        # Load all images
        loaded_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                loaded = cv2.imread(str(img))
                if loaded is None:
                    raise FileNotFoundError(f"Image not found: {img}")
                loaded_images.append(loaded)
            else:
                loaded_images.append(img)
        
        # Extract HOG features
        if self.use_hog:
            hog_features = self.hog_extractor.extract_batch(loaded_images)
        else:
            hog_features = None
        
        # Extract ResNet features in batches
        if self.use_resnet:
            resnet_features = []
            for i in range(0, len(loaded_images), batch_size):
                batch = loaded_images[i:i+batch_size]
                batch_tensor = torch.stack([
                    self.resnet_extractor.preprocess(
                        self.resnet_extractor._to_pil(img)
                    ) for img in batch
                ]).to(self.device)
                
                with torch.no_grad():
                    batch_feat = self.resnet_extractor.feature_extractor(batch_tensor)
                    batch_feat = batch_feat.view(batch_feat.size(0), -1)
                    resnet_features.append(batch_feat.cpu().numpy())
            
            resnet_features = np.vstack(resnet_features)
        else:
            resnet_features = None
        
        # Concatenate
        if hog_features is not None and resnet_features is not None:
            return np.hstack([hog_features, resnet_features])
        elif hog_features is not None:
            return hog_features
        else:
            return resnet_features


def create_image_feature_extractor(config: Dict) -> HybridImageFeatureExtractor:
    """
    Create image feature extractor from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HybridImageFeatureExtractor instance
    """
    return HybridImageFeatureExtractor(
        use_hog=config.get('use_hog', True),
        use_resnet=config.get('use_multiNet101', True),
        resnet_model='resnet101',
        hog_orientations=config.get('hog_orientations', 9)
    )
