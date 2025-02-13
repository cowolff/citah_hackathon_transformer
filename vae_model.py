import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
from skimage.transform import resize
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load the dataset from the HDF5 file
def load_dataset(h5_path, resolution=(32, 32)):
    with h5py.File(h5_path, 'r') as f:
        frames_np = np.array(f['frames'][:])
    resized_frames = np.array([resize(frame, resolution, anti_aliasing=True) for frame in frames_np])
    return resized_frames

def load_floorline_dataset(h5_path, resolution=(32, 32)):
    with h5py.File(h5_path, 'r') as f:
        floorline_np = np.array(f['floorline_masks'][:])
    resized_floorline = np.array([resize(floorline, resolution, anti_aliasing=True) for floorline in floorline_np])
    return resized_floorline

# Check image range function
def check_image_range(images):
    valid_images = []
    invalid_images = []
    for i in range(images.shape[0]):
        img = images[i]
        if img.min() == 0 and img.max() == 255:
            valid_images.append(i)
        else:
            invalid_images.append(i)
    return valid_images, invalid_images

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim=32*32, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.latent_dim = latent_dim
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar, kld_weight):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -kld_weight * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training function
def train_vae(model, dataloader, epochs=10, lr=1e-3, kld_weight=0.5, save_path=None):
    device = model.device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch in dataloader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar, kld_weight)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {train_loss / len(dataloader.dataset):.4f}")
    if save_path:
        torch.save(model.state_dict(), save_path)

def prepare_dataset(frames, valid_indices=None, normalize=True):
    if valid_indices is None:
        valid_indices, invalid_indices = check_image_range(frames)
    frames = frames[valid_indices]
    if normalize:
        frames = frames.astype(np.float32) / 255.0
    frames = torch.tensor(frames).reshape(-1, 32*32)
    return frames, valid_indices