import numpy as np
import librosa
from scipy import signal
from typing import Union, List, Tuple, Dict
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class LPCCFeatureExtractor:
    """Extract LPCC (Linear Predictive Cepstral Coefficients) features"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 n_lpcc: int = 13,
                 n_fft: int = 2048,
                 hop_length: int = 512):
        """
        Initialize LPCC extractor
        
        Args:
            sample_rate: Sample rate
            n_lpcc: Number of LPCC coefficients
            n_fft: FFT size
            hop_length: Hop length
        """
        self.sample_rate = sample_rate
        self.n_lpcc = n_lpcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract LPCC features
        
        Args:
            audio: Audio waveform
            
        Returns:
            LPCC features of shape (T, n_lpcc)
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)  # (n_fft//2+1, T)
        
        # LPC analysis for each frame
        lpcc_features = []
        
        for t in range(magnitude.shape[1]):
            frame = magnitude[:, t]
            
            # Apply window
            window = signal.hamming(len(frame))
            frame = frame * window
            
            # LPC analysis
            try:
                # Durbin-Levinson algorithm
                acf = np.correlate(frame, frame, mode='full')
                acf = acf[len(acf)//2:]
                acf /= acf[0]
                
                # LPC coefficients
                lpc_coeff = self._lpc(frame, self.n_lpcc)
                
                # Convert to cepstral coefficients
                lpcc = self._lpc_to_cepstral(lpc_coeff)
                lpcc_features.append(lpcc)
            except:
                lpcc_features.append(np.zeros(self.n_lpcc))
        
        return np.array(lpcc_features)
    
    @staticmethod
    def _lpc(signal_frame: np.ndarray, order: int) -> np.ndarray:
        """Compute LPC coefficients using Durbin-Levinson"""
        # Autocorrelation
        acf = np.correlate(signal_frame, signal_frame, mode='full')
        acf = acf[len(acf)//2:]
        acf /= (acf[0] + 1e-9)
        
        # Durbin-Levinson recursion
        lpc = np.zeros(order + 1)
        lpc[0] = 1.0
        
        error = acf[0]
        
        for i in range(order):
            delta = np.dot(acf[1:i+1], lpc[i:0:-1])
            alpha = -(acf[i+1] - delta) / (error + 1e-9)
            
            lpc_new = np.zeros(order + 1)
            lpc_new[:i+1] = lpc[:i+1]
            lpc_new[i+1] = alpha
            lpc_new[i+1:] = lpc_new[i+1:] + alpha * lpc[i::-1]
            
            lpc = lpc_new
            error = error * (1 - alpha ** 2)
        
        return lpc[1:]  # Return without 1.0 at beginning
    
    @staticmethod
    def _lpc_to_cepstral(lpc_coeff: np.ndarray, num_cepstral: int = 13) -> np.ndarray:
        """Convert LPC coefficients to cepstral coefficients"""
        cepstral = np.zeros(num_cepstral)
        
        for n in range(num_cepstral):
            if n == 0:
                cepstral[0] = np.log(1.0 / (1.0 - np.sum(lpc_coeff)))
            else:
                sum_term = sum((k + 1) / (n) * lpc_coeff[k] * cepstral[n-k-1] 
                              for k in range(n) if k < len(lpc_coeff))
                if n <= len(lpc_coeff):
                    cepstral[n] = lpc_coeff[n-1] + sum_term
                else:
                    cepstral[n] = sum_term
        
        return cepstral[:num_cepstral]


class MFCCFeatureExtractor:
    """Extract MFCC (Mel-Frequency Cepstral Coefficients) features using librosa"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 n_mfcc: int = 13,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128):
        """
        Initialize MFCC extractor
        
        Args:
            sample_rate: Sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT size
            hop_length: Hop length
            n_mels: Number of mel bands
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features
        
        Args:
            audio: Audio waveform or path
            
        Returns:
            MFCC features of shape (n_mfcc, T)
        """
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio, _ = librosa.load(str(audio), sr=self.sample_rate)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Transpose to (T, n_mfcc)
        return mfcc.T  # (T, n_mfcc)
    
    def extract_with_deltas(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC with delta and delta-delta features
        
        Args:
            audio: Audio waveform
            
        Returns:
            Combined features of shape (T, 3*n_mfcc)
        """
        mfcc = self.extract(audio)  # (T, n_mfcc)
        
        # Compute deltas
        delta = librosa.feature.delta(mfcc.T).T  # (T, n_mfcc)
        delta_delta = librosa.feature.delta(mfcc.T, order=2).T  # (T, n_mfcc)
        
        # Concatenate
        return np.hstack([mfcc, delta, delta_delta])  # (T, 3*n_mfcc)


class HybridAudioFeatureExtractor:
    """Extract features using both LPCC and MFCC"""
    
    def __init__(self,
                 use_lpcc: bool = True,
                 use_mfcc: bool = True,
                 sample_rate: int = 16000,
                 n_lpcc: int = 13,
                 n_mfcc: int = 13,
                 n_fft: int = 2048,
                 hop_length: int = 512):
        """
        Initialize hybrid audio feature extractor
        
        Args:
            use_lpcc: Whether to use LPCC features
            use_mfcc: Whether to use MFCC features
            sample_rate: Sample rate
            n_lpcc: Number of LPCC coefficients
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT size
            hop_length: Hop length
        """
        self.use_lpcc = use_lpcc
        self.use_mfcc = use_mfcc
        self.sample_rate = sample_rate
        
        if use_lpcc:
            self.lpcc_extractor = LPCCFeatureExtractor(
                sample_rate=sample_rate,
                n_lpcc=n_lpcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
        else:
            self.lpcc_extractor = None
        
        if use_mfcc:
            self.mfcc_extractor = MFCCFeatureExtractor(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
        else:
            self.mfcc_extractor = None
    
    def extract(self, audio: Union[np.ndarray, str]) -> np.ndarray:
        """
        Extract hybrid audio features
        
        Args:
            audio: Audio waveform or path
            
        Returns:
            Concatenated feature vector of shape (T, D)
        """
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio, _ = librosa.load(str(audio), sr=self.sample_rate)
        
        features = []
        
        # LPCC features
        if self.use_lpcc:
            lpcc_feat = self.lpcc_extractor.extract(audio)  # (T, n_lpcc)
            features.append(lpcc_feat)
        
        # MFCC features
        if self.use_mfcc:
            mfcc_feat = self.mfcc_extractor.extract(audio)  # (T, n_mfcc)
            features.append(mfcc_feat)
        
        # Concatenate
        if features:
            return np.hstack(features)  # (T, D)
        else:
            raise ValueError("At least one feature extractor must be enabled")
    
    def extract_batch(self, audio_list: List[Union[np.ndarray, str]]) -> List[np.ndarray]:
        """
        Extract features for batch of audio files
        
        Args:
            audio_list: List of audio waveforms or paths
            
        Returns:
            List of feature arrays
        """
        return [self.extract(audio) for audio in audio_list]


class SpectrogramExtractor:
    """Extract spectrogram-based features"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128):
        """Initialize spectrogram extractor"""
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def extract_mel_spectrogram(self, audio: Union[np.ndarray, str]) -> np.ndarray:
        """Extract mel-scale spectrogram"""
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio, _ = librosa.load(str(audio), sr=self.sample_rate)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db.T  # (T, n_mels)
    
    def extract_stft(self, audio: Union[np.ndarray, str]) -> np.ndarray:
        """Extract STFT magnitude"""
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio, _ = librosa.load(str(audio), sr=self.sample_rate)
        
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        magnitude = np.abs(stft)
        
        return magnitude.T  # (T, n_fft//2 + 1)


def create_audio_feature_extractor(config: Dict) -> HybridAudioFeatureExtractor:
    """
    Create audio feature extractor from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HybridAudioFeatureExtractor instance
    """
    return HybridAudioFeatureExtractor(
        use_lpcc=config.get('use_lpcc', True),
        use_mfcc=config.get('use_mfcc', True),
        sample_rate=config.get('sample_rate', 16000),
        n_lpcc=config.get('n_mfcc', 13),
        n_mfcc=config.get('n_mfcc', 13),
        n_fft=config.get('n_fft', 2048),
        hop_length=config.get('hop_length', 512)
    )
