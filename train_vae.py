import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from vae_model import VAE, load_dataset, load_floorline_dataset
from vae_model import prepare_dataset, train_vae
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.mps.is_available() else "cpu")

h5_path = 'pikk_hackathon/frames_data.h5'

frames = load_dataset(h5_path)
frames, valid_indices = prepare_dataset(frames)
dataset = TensorDataset(frames)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
vae = VAE().to(device)
train_vae(vae, dataloader, epochs=10, save_path="models/vae_full.pth")


# Create DataLoader for floorline
floorline_frames = load_floorline_dataset(h5_path)
floorline_frames, _ = prepare_dataset(floorline_frames, valid_indices, normalize=False)
dataset = TensorDataset(floorline_frames)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
vae = VAE().to(device)
train_vae(vae, dataloader, epochs=10, kld_weight=0.5, save_path="models/vae_floorline.pth")

# Function to test and plot original vs reconstructed images
def test_and_plot(model, dataloader, title="", num_images=10):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_images:
                break
            batch = batch[0].to(device)
            recon_batch, _, _ = model(batch)
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(batch[0].cpu().reshape(32, 32), cmap='gray')
            axes[0].set_title('Original')
            axes[1].imshow(recon_batch[0].cpu().reshape(32, 32), cmap='gray')
            axes[1].set_title('Reconstructed')
            plt.savefig("figures/recon_image_{}_{}.png".format(title, i))

# Test and plot images
test_and_plot(vae, dataloader, title="frames")
test_and_plot(vae, dataloader, title="floorline")