# Unsupervised Music Clustering using Variational Autoencoders

## Overview
This project implements an unsupervised music clustering system using a Variational Autoencoder (VAE).
The goal is to learn compact latent representations from audio features and perform clustering in the
learned latent space to identify structural similarities between music tracks.

The pipeline is fully reproducible and designed to be easily extensible with additional modalities
(e.g., lyrics, metadata) or alternative clustering algorithms.

---

## Methodology

### Audio Feature Extraction
- Audio tracks are processed using MFCC (Mel-Frequency Cepstral Coefficients).
- Each track is converted into a fixed-length feature vector for model input.

### Variational Autoencoder (VAE)
- A fully connected VAE is used to learn low-dimensional latent representations.
- Latent dimension: 16
- The VAE is trained in an unsupervised manner using reconstruction loss and KL divergence.

### Clustering
- K-Means clustering is applied to the learned latent vectors.
- The number of clusters is set to 10.
- Clustering quality is evaluated using standard unsupervised metrics.

---

## Results

### Quantitative Metrics
- Silhouette Score: 0.2267
- Calinski–Harabasz Index: 452.61

These results indicate meaningful structure in the learned latent space.

### Hybrid Audio + Lyrics Extension
An extended experiment was conducted by combining VAE-based audio latent representations
with TF-IDF features extracted from song lyrics. The hybrid feature space was evaluated
using multiple clustering algorithms, including K-Means, Agglomerative Clustering, and DBSCAN.

The hybrid representation improved clustering quality, as reflected by higher Silhouette
Scores and lower Davies–Bouldin Index values compared to audio-only clustering.

### Advanced Extension: Beta-VAE (Hard Task)
An advanced experiment was conducted using a Beta-VAE to encourage disentangled latent
representations by increasing the weight of the KL-divergence term. While clustering
metrics were slightly lower compared to the standard VAE, the Beta-VAE provides improved
interpretability of latent factors, highlighting the trade-off between disentanglement
and clustering compactness.


### Visualization
A t-SNE visualization of the VAE latent space is provided to illustrate cluster separation.

The visualization can be found at:
results/tsne_latent_space.png
---

## Conclusion
This project demonstrates that Variational Autoencoders can effectively learn compact latent
representations from audio features, enabling meaningful unsupervised clustering of music tracks.
The current implementation provides a strong foundation that can be extended with multimodal
features such as lyrics or metadata for richer music analysis.

