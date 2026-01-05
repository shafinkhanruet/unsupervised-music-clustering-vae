import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, audio_dir, max_len=130):
        self.audio_files = []
        self.max_len = max_len

        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.endswith(".wav"):
                    self.audio_files.append(os.path.join(root, file))

        print(f"Found {len(self.audio_files)} audio files")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        while True:
            file_path = self.audio_files[idx]
            try:
                y, sr = librosa.load(file_path, sr=22050)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

                if mfcc.shape[1] < self.max_len:
                    pad_width = self.max_len - mfcc.shape[1]
                    mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
                else:
                    mfcc = mfcc[:, :self.max_len]

                mfcc = mfcc.flatten()
                return torch.tensor(mfcc, dtype=torch.float32)

            except Exception:
                idx = (idx + 1) % len(self.audio_files)
