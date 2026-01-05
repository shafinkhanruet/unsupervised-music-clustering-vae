import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import MusicDataset
from vae import VAE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Paths
AUDIO_DIR = "../data/audio"
LYRICS_PATH = "../data/lyrics/lyrics.csv"
MODEL_PATH = "../results/vae_model.pth"

# -----------------------
# Load audio latent features
# -----------------------
dataset = MusicDataset(AUDIO_DIR)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

input_dim = len(dataset[0])

model = VAE(input_dim=input_dim, latent_dim=16)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

audio_latents = []

with torch.no_grad():
    for batch in dataloader:
        _, mu, _ = model(batch)
        audio_latents.append(mu.numpy())

audio_latents = np.vstack(audio_latents)

print("Audio latent shape:", audio_latents.shape)

# -----------------------
# Load and process lyrics
# -----------------------
lyrics_df = pd.read_csv(LYRICS_PATH)

vectorizer = TfidfVectorizer(max_features=50)
lyrics_features = vectorizer.fit_transform(lyrics_df["lyrics"]).toarray()

print("Lyrics feature shape:", lyrics_features.shape)

# -----------------------
# Match sizes (simple repetition if needed)
# -----------------------
if lyrics_features.shape[0] < audio_latents.shape[0]:
    repeat_factor = audio_latents.shape[0] // lyrics_features.shape[0] + 1
    lyrics_features = np.tile(lyrics_features, (repeat_factor, 1))

lyrics_features = lyrics_features[:audio_latents.shape[0]]

# -----------------------
# Combine audio + lyrics
# -----------------------
hybrid_features = np.hstack([audio_latents, lyrics_features])

scaler = StandardScaler()
hybrid_features = scaler.fit_transform(hybrid_features)

print("Hybrid feature shape:", hybrid_features.shape)

# -----------------------
# Clustering experiments
# -----------------------

# K-Means
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans_labels = kmeans.fit_predict(hybrid_features)

# Agglomerative
agg = AgglomerativeClustering(n_clusters=10)
agg_labels = agg.fit_predict(hybrid_features)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
db_labels = dbscan.fit_predict(hybrid_features)

# -----------------------
# Evaluation
# -----------------------
print("\nEvaluation Metrics (Hybrid Features):")

print("K-Means Silhouette:", silhouette_score(hybrid_features, kmeans_labels))
print("K-Means Davies-Bouldin:", davies_bouldin_score(hybrid_features, kmeans_labels))

print("Agglomerative Silhouette:", silhouette_score(hybrid_features, agg_labels))
print("Agglomerative Davies-Bouldin:", davies_bouldin_score(hybrid_features, agg_labels))

# DBSCAN may label noise as -1
if len(set(db_labels)) > 1 and -1 not in set(db_labels):
    print("DBSCAN Silhouette:", silhouette_score(hybrid_features, db_labels))
    print("DBSCAN Davies-Bouldin:", davies_bouldin_score(hybrid_features, db_labels))
else:
    print("DBSCAN produced noise or a single cluster (expected on some datasets)")
