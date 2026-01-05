import torch
from torch.utils.data import DataLoader
from dataset import MusicDataset
from vae import VAE
import torch.nn.functional as F
import os

# Paths
AUDIO_DIR = "../data/audio"
MODEL_PATH = "../results/vae_model.pth"

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LATENT_DIM = 16
LEARNING_RATE = 1e-3

# Dataset & DataLoader
dataset = MusicDataset(AUDIO_DIR)
input_dim = len(dataset[0])

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

# Save model
os.makedirs("../results", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved to", MODEL_PATH)
