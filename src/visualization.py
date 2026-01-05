import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import MusicDataset
from vae import VAE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# Paths
AUDIO_DIR = "../data/audio"
MODEL_PATH = "../results/vae_model.pth"
OUTPUT_DIR = "../results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
dataset = MusicDataset(AUDIO_DIR)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

input_dim = len(dataset[0])

# Load trained VAE
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

print("Latent vectors shape:", latent_vectors.shape)

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
latent_2d = tsne.fit_transform(latent_vectors)

plt.figure(figsize=(8, 6))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], s=10, alpha=0.7)
plt.title("t-SNE Visualization of VAE Latent Space")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

output_path = os.path.join(OUTPUT_DIR, "tsne_latent_space.png")
plt.savefig(output_path, dpi=300)
plt.close()

print("Saved t-SNE plot to:", output_path)
