import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import MusicDataset
from vae import VAE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os

# Paths
AUDIO_DIR = "../data/audio"
MODEL_PATH = "../results/vae_model.pth"

# Load dataset
dataset = MusicDataset(AUDIO_DIR)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

input_dim = len(dataset[0])

# Load trained VAE
model = VAE(input_dim=input_dim, latent_dim=16)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Extract latent features
latent_vectors = []

with torch.no_grad():
    for batch in dataloader:
        _, mu, _ = model(batch)
        latent_vectors.append(mu.numpy())

latent_vectors = np.vstack(latent_vectors)

print("Latent feature shape:", latent_vectors.shape)

# K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(latent_vectors)

# Evaluation metrics
sil_score = silhouette_score(latent_vectors, labels)
ch_score = calinski_harabasz_score(latent_vectors, labels)

print("Silhouette Score:", sil_score)
print("Calinski-Harabasz Index:", ch_score)
