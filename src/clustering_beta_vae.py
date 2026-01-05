import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import MusicDataset
from vae import VAE

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Paths
AUDIO_DIR = "../data/audio"
MODEL_PATH = "../results/beta_vae_model.pth"

# Load dataset
dataset = MusicDataset(AUDIO_DIR)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

input_dim = len(dataset[0])

# Load Beta-VAE model
model = VAE(input_dim=input_dim, latent_dim=16)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Extract latent vectors
latent_vectors = []

with torch.no_grad():
    for batch in dataloader:
        _, mu, _ = model(batch)
        latent_vectors.append(mu.numpy())

latent_vectors = np.vstack(latent_vectors)

print("Beta-VAE latent shape:", latent_vectors.shape)

# K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(latent_vectors)

# Metrics
sil = silhouette_score(latent_vectors, labels)
ch = calinski_harabasz_score(latent_vectors, labels)
db = davies_bouldin_score(latent_vectors, labels)

print("Beta-VAE Silhouette Score:", sil)
print("Beta-VAE Calinski-Harabasz Index:", ch)
print("Beta-VAE Davies-Bouldin Index:", db)
