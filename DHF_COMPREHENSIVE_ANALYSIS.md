# Deep Hybrid Fusion (DHF) Project - Comprehensive Analysis

## Executive Summary

The Deep Hybrid Fusion project is a comprehensive multimodal machine learning framework for sentiment analysis and emotion recognition that combines multiple fusion strategies and feature extraction techniques across text, audio, and visual modalities.

---

## 1. ARCHITECTURE OVERVIEW

### 1.1 Three-Stage Architecture

The DHF architecture (as shown in `images/Architecture.png`) consists of three main stages:

```
┌─────────────────────────┐
│   Video Feature         │
│   Extraction            │
├─────────────────────────┤
│   Modality              │
│   Representation        │
├─────────────────────────┤
│   Deep Hybrid           │
│   Fusion                │
└─────────────────────────┘
```

---

## 2. STAGE 1: VIDEO FEATURE EXTRACTION

### 2.1 Text Feature Extraction
**Method:** Enhanced BERT Model
- Utilizes pre-trained BERT (Bidirectional Encoder Representations from Transformers)
- Extracts contextual word embeddings
- Used across multiple implementations (MISA, contextual-multimodal-fusion)
- Dimensionality: typically 768-dimensional embeddings reduced to hidden_size
- Fine-tuned for sentiment/emotion classification tasks

### 2.2 Image/Visual Feature Extraction
**Methods:** HOG and MultiNet-101
- **HOG (Histogram of Oriented Gradients):**
  - Traditional computer vision feature descriptor
  - Captures edge and gradient information
  - Robust to lighting changes
  
- **MultiNet-101 (ResNet-101 variant):**
  - Deep CNN for visual feature extraction
  - Pre-trained on ImageNet
  - Provides high-level semantic visual representations
  - Typically outputs 2048-dimensional vectors that are further reduced

### 2.3 Audio Feature Extraction
**Methods:** LPCC and Librosa
- **LPCC (Linear Prediction Cepstral Coefficients):**
  - Speech-specific feature extraction
  - Captures spectral characteristics of audio
  - Computationally efficient
  
- **Librosa:**
  - Python library for music and audio analysis
  - Extracts MFCCs (Mel-frequency cepstral coefficients)
  - Provides temporal and spectral features
  - Common feature dimensions: 13-40 coefficients

### 2.4 Feature Dimensionality Summary
| Modality | Feature Type | Extraction Method | Typical Output Dimension |
|----------|-------------|------------------|------------------------|
| Text | Embeddings | BERT | 768 → hidden_size |
| Video | HOG | Traditional | 1024-2048 |
| Video | CNN | MultiNet-101 | 2048 |
| Audio | LPCC | Signal processing | 13-40 |
| Audio | MFCC | Librosa | 13-40 |

---

## 3. STAGE 2: MODALITY REPRESENTATION

### 3.1 Domain Unchanged Encoder
**Mathematical Formulation:**
$$h_m^c = E_c(a_m; \theta^c)$$

Where:
- $h_m^c$ = context-preserved representation for modality m
- $E_c$ = domain unchanged/common encoder
- $a_m$ = input features from modality m
- $\theta^c$ = shared encoder parameters

**Purpose:** 
- Maintains modality-specific characteristics while extracting common representations
- Uses loss function: $L_{sim}$ (similarity loss)
- Ensures features from different modalities share common semantic space

**Implementation Pattern (from contextual architectures):**
```python
# Project raw features to common space
project_t = nn.Sequential(
    nn.Linear(in_features=768, out_features=hidden_size),
    activation(),
    nn.LayerNorm(hidden_size)
)
```

### 3.2 Domain Precise Encoder
**Mathematical Formulation:**
$$h_m^p = E_p(a_m; \theta_m^p)$$

Where:
- $h_m^p$ = modality-specific precise representation
- $E_p$ = domain precise/private encoder
- $\theta_m^p$ = modality-specific parameters

**Purpose:**
- Preserves modality-unique information
- Uses loss function: $L_{diff}$ (difference loss)
- Captures discriminative features specific to each modality

**Implementation Pattern (from MISA):**
```python
# Private modality encoders with Sigmoid activation
self.private_t = nn.Sequential(
    nn.Linear(in_features=hidden_size, out_features=hidden_size),
    nn.Sigmoid()
)
self.private_v = nn.Sequential(
    nn.Linear(in_features=hidden_size, out_features=hidden_size),
    nn.Sigmoid()
)
self.private_a = nn.Sequential(
    nn.Linear(in_features=hidden_size, out_features=hidden_size),
    nn.Sigmoid()
)
```

### 3.3 Unified Representation
**Combination Formula:**
$$h_m = h_m^c + h_m^p$$

The final modality representation combines both common and precise representations through element-wise addition or concatenation.

### 3.4 Encoder Dimensionalities (from code analysis)

**Text Encoder:**
- Input: BERT embeddings (768)
- Projection: hidden_size (typically 128-256)
- RNN (Bidirectional LSTM/GRU):
  - 2 layers with bidirectional processing
  - Output: hidden_size × 4 (accounting for bidirectional stacking)

**Visual Encoder:**
- Input: CNN features (2048)
- RNN: 2-layer Bidirectional LSTM/GRU
- Output: hidden_size × 4

**Audio Encoder:**
- Input: LPCC/MFCC features (13-40)
- RNN: 2-layer Bidirectional LSTM/GRU
- Output: hidden_size × 4

---

## 4. STAGE 3: DEEP HYBRID FUSION

### 4.1 Fusion Input
**Concatenated representation:**
$$z = [h_m^c + h_m^p, h_m^c + h_m^p, h_m^c + h_m^p]_{text, audio, video}$$

Input dimension to fusion = 3 × hidden_size × 4 = 12 × hidden_size

### 4.2 Transformer-MultiHead Attention (M)
**Purpose:** 
- Cross-modal interaction and attention
- Multi-head attention mechanism for capturing different aspects of modality interactions

**Architecture (from code patterns):**
```python
# Multi-head attention across modalities
class TransformerMultiNet(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        # Implementation uses standard transformer mechanism
        # Processes concatenated modality representations
```

**Key Features:**
- Learns multiple attention heads for different interaction patterns
- Allows each modality to attend to others selectively
- Number of heads typically: 4-16

### 4.3 Fusion Layer
**Input:** Attention-weighted multimodal features
**Parameters:**
- Input: 12 × hidden_size (from three modalities with context and private)
- Intermediate: 6 × hidden_size
- Uses dropout for regularization

**Implementation (from MISA):**
```python
self.fusion = nn.Sequential()
self.fusion.add_module('fusion_layer_1', 
    nn.Linear(in_features=self.config.hidden_size*6, 
              out_features=self.config.hidden_size*3))
self.fusion.add_module('fusion_layer_1_dropout', 
    nn.Dropout(dropout_rate))
```

### 4.4 Decoder Layer
**Mathematical Formulation:**
$$\hat{y} = \text{Decoder}(h_m^c + h_m^p)$$

**Input:** Fused multimodal representation
**Output:** Classification logits/probabilities

**Architecture:**
```python
Decoder = LinearLayer(hidden_size*3 → hidden_size) 
          → Activation → Dropout
          → LinearLayer(hidden_size → num_classes)
```

### 4.5 Output Layer
**Final Prediction:**
$$\hat{y} = (h^{out}, \theta^{out})$$

Where:
- $h^{out}$ = output feature representation
- $\theta^{out}$ = output parameters (logits)

**Loss Function:** $L_{task}$ (task-specific classification loss)

---

## 5. COMPLETE TRAINING OBJECTIVE

### 5.1 Multi-Objective Loss Function

$$L_{total} = L_{task} + \lambda_1 L_{sim} + \lambda_2 L_{diff}$$

Where:
- **$L_{task}$**: Classification/regression loss (CrossEntropy, MSE, etc.)
- **$L_{sim}$**: Similarity loss encouraging common representation alignment
- **$L_{diff}$**: Difference loss enforcing modality-specific information orthogonality
- **$\lambda_1, \lambda_2$**: Loss weight hyperparameters

### 5.2 Alternative Fusion Approaches in Codebase

**Tensor Fusion Network (TFN):**
```
Outer product of representations → Low-rank factorization → Dense layers
```
**Low-Rank Multimodal Fusion (LMF):**
```
Low-rank factorization: 
Z = Σ(rank_k) (Audio_factor × Video_factor × Text_factor)
```

---

## 6. MULTIMODAL FUSION IMPLEMENTATIONS

### 6.1 bc-LSTM (Bidirectional Context-LSTM)
**Architecture:**
- 2-layer Bidirectional LSTM for each modality
- Context modeling through sequential processing
- Per-utterance predictions in conversational data
- Unimodal and multimodal variants

**Input Processing:**
```python
lstm = Bidirectional(LSTM(300, activation='tanh', 
                          return_sequences=True, dropout=0.4))
lstm = Bidirectional(LSTM(300, activation='tanh', 
                          return_sequences=True, dropout=0.4))
output = TimeDistributed(Dense(num_classes, activation='softmax'))
```

### 6.2 Contextual-Attention-Based-LSTM
**Key Components:**
- Multiple attention mechanisms at different levels
- Contextual information fusion
- Utterance-level and dialogue-level interactions
- Supports both MOSI, MOSEI, IEMOCAP datasets

**Attention Types:**
- Intra-modal attention (within modality)
- Inter-modal attention (between modalities)
- Context-aware attention

### 6.3 Contextual-Multimodal-Fusion
**Features:**
- Inter-modal contextual attention
- Dynamic feature fusion based on utterance context
- Multimodal sentiment analysis
- MOSI dataset focus

### 6.4 MISA (Modality-Invariant and -Specific Representations)
**Architecture Pattern:**
```
Input Features
    ↓
[Bidirectional RNN layers] ← Shared components
    ↓
Projection to common space
    ↓
Shared Encoder + Private Encoders
    ↓
Reconstruction & Adversarial Loss
    ↓
Fusion layer
    ↓
Classification
```

**Novel Components:**
- Shared-private representation learning
- Modality-invariant features (shared encoder)
- Modality-specific features (private encoders)
- Reconstruction loss for robustness
- Adversarial discriminator for domain alignment

**Hyperparameters:**
```python
config = {
    'embedding_size': 300,
    'visual_size': 47,
    'acoustic_size': 74,
    'hidden_size': 128,
    'num_classes': 2-3 (depending on dataset),
    'dropout': 0.2,
    'rnncell': 'lstm' or 'gru'
}
```

### 6.5 Tensor Fusion Networks (TFN)
**Core Mechanism:** Outer Product Fusion
```
Fusion_z = [Audio_h ⊗ Video_h ⊗ Text_h] + bias
```

**Implementation:**
```python
class TFN(nn.Module):
    # Audio, Video, Text subnets with different processing
    # Outer product creates comprehensive feature space
    # Post-fusion dense layers for prediction
```

**Advantages:**
- Captures all pairwise and higher-order interactions
- Comprehensive feature space representation

### 6.6 Low-Rank Multimodal Fusion (LMF)
**Low-Rank Factorization:**
```
Z = Σ_{k=1}^{rank} (u_k^audio ⊗ v_k^video ⊗ w_k^text)
```

**Implementation:**
```python
class LMF(nn.Module):
    # Rank-based factorization
    # Reduces computational complexity vs TFN
    # Similar expressiveness with lower dimension
```

### 6.7 MELD (Multimodal EmotionLines Dataset)
**Baseline Architecture:**
- bc-LSTM backbone
- Supports emotion (7 classes) and sentiment (3 classes) classification
- Multi-party conversation handling
- Audio features: openSMILE (1611/1422-dim)

### 6.8 MUStARD (Multimodal Sarcasm Detection)
**Feature Components:**
- Text: BERT embeddings
- Audio: Extracted features with librosa
- Visual: C3D or I3D CNN features
- Sarcasm detection task

### 6.9 Hierarchical Fusion (hfusion)
**Structure:**
- Modality-specific feature extraction
- Hierarchical combination of modalities
- Context modeling at multiple levels
- Sentiment analysis focus

---

## 7. DATASETS AND STRUCTURE

### 7.1 Dataset Specifications

#### CMU-MOSI (Multimodal Opinion Sentiment Intensity)
**Task:** Sentiment analysis
**Classes:** 2 (Positive/Negative)
**Characteristics:**
- 23.1 hours of video
- 2199 utterances from 1000+ videos
- Continuous sentiment scores
- Pre-processed into pickle files

**Feature Files:**
```
dataset/mosi/raw/
├── audio_2way.pickle
├── text_2way.pickle
└── video_2way.pickle
```

**Feature Dimensions:**
```python
Audio: (n_samples, seq_len, 74) or (n_samples, seq_len, 81)
Text: (n_samples, seq_len, 300)
Video: (n_samples, seq_len, 35)
```

**Data Processing:**
- Train/Val/Test split with pickle dictionaries
- Indexed by dialogue_id_utterance_id (e.g., '0_0')
- Shape: (33, 300) for 600-dim tokens with max 33 utterances

#### CMU-MOSEI (Multimodal Opinion Sentiment and Emotion Intensity)
**Task:** Sentiment and emotion analysis
**Classes:** 3 (Positive/Neutral/Negative)
**Characteristics:**
- 35+ hours of video
- 23,000+ utterances
- 1000+ speakers
- More diverse than MOSI

**Data Structure:**
```
dataset/mosei/raw/
├── audio_3way.pickle
├── text_3way.pickle
└── video_3way.pickle
```

#### IEMOCAP (Interactive Emotional Dyadic Motion Capture Database)
**Task:** Emotion recognition
**Classes:** 6 (happy/sad/neutral/angry/excited/frustrated)
**Characteristics:**
- 12 hours of multimodal data
- 2-person dialogues
- Acted emotions
- Dyadic conversation structure

**Raw Features:**
```
dataset/iemocap/raw/IEMOCAP_features_raw.pkl
```

**Feature Dimensions:**
```python
Audio: Mel-frequency features
Video: 3D CNN features
Text: BERT embeddings
```

#### MELD (Multimodal EmotionLines Dataset)
**Task:** Emotion and sentiment in conversations
**Emotion Classes:** 7 (Anger, Disgust, Sadness, Joy, Neutral, Surprise, Fear)
**Sentiment Classes:** 3 (Positive, Negative, Neutral)
**Characteristics:**
- 1400+ dialogues from Friends TV series
- 13,000+ utterances
- Multi-party conversations (2-6 speakers)
- Video clips with audio and text

**Data Files:**
```
MELD/data/
├── train_sent_emo.csv
├── val_sent_emo.csv
├── test_sent_emo.csv
├── pickles/
│   ├── text_glove_CNN_emotion.pkl
│   ├── audio_embeddings_feature_selection_emotion.pkl
│   ├── text_emotion.pkl (600-dim contextual)
│   ├── audio_emotion.pkl (300/600-dim)
│   └── bimodal_sentiment.pkl (600-dim)
└── MELD/  (video clips)
```

**Audio Feature Dimensions:**
- openSMILE features: 1611-dimensional (emotion), 1422-dimensional (sentiment)
- L2-based feature selection via SVM
- Indexed as: `train_audio_emb['0_0'].shape = (1611,)`

**Contextual Feature Dimensions:**
```python
Text: (max_utt, 600)        # 33 max utterances per dialogue
Audio: (max_utt, 300/600)
Video: (max_utt, 2048)
```

#### MUStARD (Multimodal Sarcasm Detection Dataset)
**Task:** Sarcasm detection
**Classes:** 2 (Sarcastic/Non-sarcastic)
**Characteristics:**
- From TV shows: Friends, The Big Bang Theory, Golden Girls
- 690 utterances with context
- Contextual clips required

**Data Structure:**
```
MUStARD/data/
├── sarcasm_data.json
├── bert-input.txt
├── audio_features.p
├── glove_full_dict.p
└── split_indices.p
```

### 7.2 Data File Formats

**Pickle Files (Binary):**
```python
# Typical structure
data = {
    'train': [features, labels],
    'val': [features, labels],
    'test': [features, labels]
}
# Or indexed by dialogue_id: data['0_0'] for dialogue 0, utterance 0
```

**CSV Format:**
```
Utterance | Speaker | Dialogue_ID | Utterance_ID | Emotion | Sentiment
```

**Processed Features (Numpy arrays):**
```
Shape: (n_samples, sequence_length, feature_dim)
- n_samples: number of dialogues/videos
- sequence_length: max utterances (padded)
- feature_dim: feature vector dimension
```

### 7.3 Dataset Summary Table

| Dataset | Task | Classes | Size | Modalities | Key Feature |
|---------|------|---------|------|-----------|------------|
| MOSI | Sentiment | 2 | 2199 utt | A,V,T | Continuous scores |
| MOSEI | Sentiment/Emotion | 3 | 23000+ utt | A,V,T | Large-scale |
| IEMOCAP | Emotion | 6 | 10000+ utt | A,V,T | Dyadic, acted |
| MELD | Emotion/Sentiment | 7/3 | 13000+ utt | A,V,T | Multi-party dialogues |
| MUStARD | Sarcasm | 2 | 690 | A,V,T | Context-dependent |

---

## 8. REQUIRED PYTHON LIBRARIES AND DEPENDENCIES

### 8.1 Core ML/DL Frameworks

**PyTorch-Based Implementations (MISA, LMF, TFN):**
```
torch >= 1.3.1
torchvision >= 0.4.2
pytorch-lightning (optional)
```

**Keras/TensorFlow-Based Implementations (hfusion, bc-LSTM, contextual models):**
```
tensorflow >= 1.7 or >= 2.0
keras >= 2.0
```

### 8.2 NLP and Transformers

```
transformers >= 2.8.0  # BERT models
bert-pytorch
nltk >= 3.4
spacy >= 2.1
gensim >= 3.8
```

### 8.3 Audio and Signal Processing

```
librosa >= 0.7.0       # Audio feature extraction
scipy >= 1.3.0         # Signal processing
scikit-learn >= 0.21   # Machine learning utilities
numpy >= 1.17.3        # Numerical computing
```

### 8.4 Computer Vision

```
opencv-python >= 3.4
pillow >= 6.2.1
scikit-image >= 0.16   # For HOG features
```

### 8.5 Data Processing and Utilities

```
pandas >= 0.24         # Data manipulation
scikit-learn >= 0.21   # ML utilities including feature selection
pickle (built-in)      # Data serialization
h5py >= 2.9           # HDF5 file handling (for saving models)
```

### 8.6 Evaluation and Visualization

```
matplotlib >= 3.1
seaborn >= 0.9
scikit-metrics        # Classification metrics
```

### 8.7 Environment Management

**Conda environment (from MISA):**
```
python >= 3.7.5
cudatoolkit >= 10.1
numpy >= 1.17.3
```

### 8.8 Complete Requirements Example

**For PyTorch-based models (MISA):**
```
torch==1.3.1
numpy==1.17.3
scikit-learn
transformers>=2.8.0
librosa>=0.7.0
scipy
tqdm
```

**For Keras/TensorFlow models (hfusion, bc-LSTM):**
```
tensorflow>=1.7
keras>=2.0
numpy
scikit-learn
scipy
librosa>=0.7.0
```

### 8.9 Optional Dependencies

```
jupyter >= 1.0         # Interactive notebooks
ipython >= 7.0
tensorboard            # Training visualization
wandb                  # Experiment tracking
```

---

## 9. IMPLEMENTATION PATTERNS AND BEST PRACTICES

### 9.1 Feature Extraction Pattern

```python
# Text: BERT-based
from transformers import BertModel, BertTokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
token_embeddings = bert_model(input_ids, attention_mask)[0]
# Shape: (batch_size, seq_len, 768)

# Audio: Librosa-based
import librosa
y, sr = librosa.load('audio.wav')
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
# Shape: (n_mfcc, time_steps)

# Video: CNN-based (ResNet-101 or C3D)
import torchvision.models as models
model = models.resnet101(pretrained=True)
features = model(video_frames)
# Shape: (batch_size, 2048)
```

### 9.2 RNN Encoding Pattern

```python
# Bidirectional LSTM/GRU encoding
class ModalityEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn1 = nn.LSTM(input_size, hidden_size, 
                           bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(2*hidden_size, hidden_size, 
                           bidirectional=True, batch_first=True)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        out1, _ = self.rnn1(x)  # (batch, seq_len, 2*hidden)
        out2, _ = self.rnn2(out1)  # (batch, seq_len, 2*hidden)
        return out2  # Final output: (batch_size, seq_len, 2*hidden_size)
```

### 9.3 Shared-Private Representation Pattern

```python
# Shared representation
self.shared = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.Sigmoid()
)

# Private representations
self.private_t = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.Sigmoid()
)

# Combination
h_shared = self.shared(projected_features)
h_private = self.private_t(projected_features)
h_final = h_shared + h_private  # Element-wise sum
```

### 9.4 Multimodal Fusion Pattern

```python
# Concatenate modality-specific representations
h_fused = torch.cat([h_text, h_audio, h_video], dim=-1)
# Shape: (batch_size, 3*hidden_size)

# Apply fusion layers
fused = self.fusion_layer(h_fused)
# Fused representation: (batch_size, hidden_size)

# Classification head
logits = self.classifier(fused)
# Output: (batch_size, num_classes)
```

### 9.5 Validation and Metrics

```python
# Common evaluation metrics
from sklearn.metrics import (accuracy_score, 
                            confusion_matrix,
                            classification_report,
                            precision_recall_fscore_support)

true_labels = []
pred_labels = []

for pred, label, mask in dataset:
    # Only evaluate non-masked positions
    valid_mask = mask == 1
    true_labels.extend(label[valid_mask])
    pred_labels.extend(pred.argmax(dim=-1)[valid_mask])

print(classification_report(true_labels, pred_labels))
print("Accuracy:", accuracy_score(true_labels, pred_labels))
```

---

## 10. KEY ARCHITECTURAL INNOVATIONS

### 10.1 Domain Unchanged vs Domain Precise Distinction
- Separates common semantic information from modality-specific characteristics
- Enables better generalization and modality-agnostic learning
- Supports robustness to missing modalities

### 10.2 Deep Hybrid Fusion
- Combines multiple fusion strategies:
  - Concatenation for feature richness
  - Attention for selective combination
  - Tensor factorization for interaction modeling
  
### 10.3 Shared-Private Representation Learning (MISA)
- Shared encoder captures modality-invariant information
- Private encoders capture modality-specific patterns
- Reconstruction loss ensures information preservation
- Adversarial loss enforces proper separation

### 10.4 Contextual Multimodal Analysis
- Leverages utterance context in dialogues
- Bidirectional LSTM captures bidirectional information flow
- Attention mechanisms focus on relevant contextual utterances

### 10.5 Task-Agnostic Framework
- Adaptable to sentiment analysis, emotion recognition, sarcasm detection
- Modular design allows component swapping
- Supports variable number of modalities and sequence lengths

---

## 11. TRAINING METHODOLOGY

### 11.1 Data Preparation

```python
# Load pickle files
with open('audio.pickle', 'rb') as f:
    train_audio, train_label, test_audio, test_label = pickle.load(f)

# Create masks for variable-length sequences
train_mask = np.zeros((len(train_data), max_len))
for i, length in enumerate(train_lengths):
    train_mask[i, :length] = 1.0

# One-hot encode labels
train_label_onehot = np.zeros((n_samples, max_len, n_classes))
for i, j, label in zip(*np.nonzero(train_label)):
    train_label_onehot[i, j, label] = 1
```

### 11.2 Training Loop Pattern

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        audio, video, text, labels, mask = batch
        
        # Forward pass through modality encoders
        h_audio = audio_encoder(audio)
        h_video = video_encoder(video)
        h_text = text_encoder(text)
        
        # Domain unchanged (shared) representations
        h_audio_shared = shared_encoder(h_audio)
        h_video_shared = shared_encoder(h_video)
        h_text_shared = shared_encoder(h_text)
        
        # Domain precise (private) representations
        h_audio_private = private_audio_encoder(h_audio)
        h_video_private = private_video_encoder(h_video)
        h_text_private = private_text_encoder(h_text)
        
        # Fusion
        h_fused = fusion([h_audio_shared + h_audio_private,
                         h_video_shared + h_video_private,
                         h_text_shared + h_text_private])
        
        # Prediction
        logits = classifier(h_fused)
        
        # Loss computation
        task_loss = criterion(logits, labels)
        sim_loss = compute_similarity_loss(h_shared_representations)
        diff_loss = compute_difference_loss(h_shared, h_private)
        
        total_loss = task_loss + lambda1*sim_loss + lambda2*diff_loss
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Validation mask applied during evaluation
        valid_pred = logits[mask == 1]
        valid_label = labels[mask == 1]
```

### 11.3 Hyperparameter Ranges

**Common Hyperparameter Ranges:**

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| hidden_size | 128-512 | Smaller for resource-constrained |
| num_layers | 1-3 | RNN layers per modality |
| dropout | 0.2-0.5 | Regularization strength |
| learning_rate | 1e-4 to 1e-3 | Adam optimizer typical |
| batch_size | 16-64 | Depends on GPU memory |
| num_epochs | 50-200 | With early stopping |
| patience (ES) | 10-20 | Early stopping patience |
| rank (LMF) | 4-16 | Factorization rank |
| num_heads (TFN) | 4-16 | For attention mechanisms |

---

## 12. PERFORMANCE METRICS AND EVALUATION

### 12.1 Classification Metrics

```python
# Accuracy (primary metric for most datasets)
accuracy = correct_predictions / total_predictions

# Precision, Recall, F1-Score
from sklearn.metrics import precision_recall_fscore_support
micro_precision, micro_recall, micro_f1, _ = \
    precision_recall_fscore_support(y_true, y_pred, average='micro')

weighted_f1 = \
    precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]

# Macro and Weighted averages
macro_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
```

### 12.2 Confusion Matrix Analysis

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
# Analyzed for class-wise performance
```

### 12.3 Dataset-Specific Metrics

**MOSI/MOSEI:**
- Accuracy for discrete classes
- Correlation for continuous sentiment scores
- MAE (Mean Absolute Error) for intensity prediction

**MELD:**
- Per-class weighted accuracy (due to class imbalance)
- Class weights: [4.0, 15.0, 15.0, 3.0, 1.0, 6.0, 3.0]

**MUStARD:**
- F1-score for binary sarcasm detection
- Precision/Recall tradeoff analysis

---

## 13. FILE ORGANIZATION AND USAGE

### 13.1 Repository Structure

```
Deep-Hybrid-Fusion/
├── bc-LSTM/                          # Context-dependent LSTM
│   ├── lstm.py                       # Model implementation
│   ├── create_data.py                # Data preprocessing
│   └── data/
│       ├── audio/, text/, video/     # Feature files
│       └── *.arff, *.csv             # Data formats
│
├── contextual-attention-based-LSTM/  # Attention mechanisms
│   ├── model.py
│   ├── run.py
│   └── dataset/
│       ├── mosi/, mosei/, iemocap/
│       └── raw/                      # Raw features
│
├── contextual-multimodal-fusion/     # Inter-modal attention
│   ├── trimodal_attention_models.py
│   └── create_data.py
│
├── MISA/                             # Modality-Invariant & Specific
│   ├── src/
│   │   ├── models.py                 # Main architecture
│   │   ├── train.py
│   │   ├── config.py
│   │   └── data_loader.py
│   └── environment.yml
│
├── Low-rank-Multimodal-Fusion/       # LMF fusion
│   ├── model.py
│   ├── train_*.py                    # Dataset-specific training
│   └── utils.py
│
├── TensorFusionNetworks/             # Tensor outer product fusion
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── MELD/                             # Multi-party conversation dataset
│   ├── baseline/
│   │   └── baseline.py
│   ├── data/
│   │   └── pickles/                  # Pre-computed features
│   └── utils/
│
├── MUStARD/                          # Sarcasm detection dataset
│   ├── data_loader.py
│   ├── config.py
│   └── visual/                       # Feature extraction scripts
│
├── hfusion/                          # Hierarchical fusion
│   ├── hfusion.py
│   └── README.md
│
└── images/
    └── Architecture.png              # DHF architecture diagram
```

### 13.2 Model Selection Guide

| Dataset | Recommended Model | Reason |
|---------|------------------|--------|
| MOSI/MOSEI | MISA or Attention-LSTM | Good balance of performance |
| IEMOCAP | Contextual-Attention-LSTM | Handles 6-class emotion well |
| MELD | bc-LSTM baseline | Multi-party conversation support |
| MUStARD | Custom (transformer-based) | Sarcasm requires context |
| Large-scale | MISA | Scalable representation learning |
| Limited data | LMF | Low-rank reduces parameters |

---

## 14. QUICK START EXAMPLES

### 14.1 MISA Model Training

```bash
# 1. Setup environment
conda env create -f MISA/environment.yml
conda activate misa-code

# 2. Download data
# Place pre-computed features in 'datasets/' folder

# 3. Train model
python MISA/src/train.py --data mosi
python MISA/src/train.py --data mosei
python MISA/src/train.py --data ur_funny
```

### 14.2 Contextual Attention LSTM

```bash
# Train with different configurations
python run.py \
    --unimodal False \       # Use all modalities
    --fusion True \          # Enable fusion
    --attention_2 True \     # Multi-level attention
    --data mosei \
    --classes 3
```

### 14.3 MELD Baseline

```bash
# Download features
# Extract to MELD/data/pickles/

# Run baseline
python MELD/baseline/baseline.py \
    -classify Sentiment \
    -modality bimodal \
    -train
```

---

## 15. SUMMARY TABLE OF ALL MODELS

| Model | Framework | Fusion Strategy | Key Innovation | Best For |
|-------|-----------|-----------------|----------------|----------|
| bc-LSTM | Keras | Concatenation | Context in dialogues | MELD, conversational |
| Attention-LSTM | Keras/PyTorch | Attention | Multi-level attention | IEMOCAP, detailed analysis |
| MISA | PyTorch | Shared-Private | Modality invariance | General purpose, robustness |
| TFN | PyTorch | Tensor outer product | All interactions | Performance-critical |
| LMF | PyTorch | Low-rank factorization | Efficiency | Resource constraints |
| hfusion | Keras | Hierarchical | Multi-level hierarchy | Structured problems |
| Contextual-multimodal | Keras | Inter-modal attention | Context interactions | MOSI, sentiment |

---

## 16. REFERENCES TO KEY FILES

- **Architecture Diagram:** [images/Architecture.png](images/Architecture.png)
- **MISA Models:** [MISA/src/models.py](MISA/src/models.py)
- **LMF Model:** [Low-rank-Multimodal-Fusion/model.py](Low-rank-Multimodal-Fusion/model.py)
- **TFN Model:** [TensorFusionNetworks/model.py](TensorFusionNetworks/model.py)
- **bc-LSTM:** [bc-LSTM/lstm.py](bc-LSTM/lstm.py)
- **MELD Baseline:** [MELD/baseline/baseline.py](MELD/baseline/baseline.py)
- **Attention Model:** [contextual-attention-based-LSTM/model.py](contextual-attention-based-LSTM/model.py)

---

**Analysis Generated:** March 22, 2026
**Project:** Deep Hybrid Fusion (DHF) for Multimodal Sentiment & Emotion Analysis

