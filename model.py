import torch
import torch.nn as nn
import librosa
import numpy as np

class AudioClassificationModel(nn.Module):
    """
    A simple 1D Convolutional Neural Network for processing extracted audio features.
    """
    def __init__(self, num_classes=3): # Assuming 3 classes (Normal, Wheeze, Crackle)
        super(AudioClassificationModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 10, 64), # Adjusted for feature layout size
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

def preprocess_audio(file_path, max_pad_len=40):
    """
    Extracts MFCC (Mel-Frequency Cepstral Coefficients) features from an audio file.
    """
    try:
        # Load audio file (resampled to 16kHz for uniform processing)
        y, sr = librosa.load(file_path, sr=16000, duration=5.0)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Normalize/Scale features across time axes
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
            
        # Take mean over time frames to produce a 1D vector of shape (40,)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Reshape to fit PyTorch Conv1d format: (Batch, Channel, Features) -> (1, 1, 40)
        tensor_data = torch.tensor(mfcc_mean, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor_data
    except Exception as e:
        print(f"Error preprocessing audio file {file_path}: {e}")
        return None