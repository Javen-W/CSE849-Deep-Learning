import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
plt.switch_backend("agg")

from models import MLP
from data import States

plot_dir = "outputs/plots/unconditional_generation"
os.makedirs(plot_dir, exist_ok=True)

# Seed
torch.manual_seed(777)

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 10_000
num_epochs = 5_000
lr = 0.001
weight_decay = 1e-4

# Create the denoiser model
mlp = MLP(input_dim=3, output_dim=2, hidden_layers=[256, 256, 256, 256]).to(device)
mse_loss = nn.MSELoss() # create the denoising (MSE) loss function

# Create your optimizer and learning rate scheduler
optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
)

num_steps = 500
dataset = States(num_steps=num_steps)
dataset.show(save_to=os.path.join(plot_dir, "original_data.png"))

train_loss_list = []
nll_list = []

def train_one_epoch():
    avg_loss = 0
    batch_idx = 0
    pbar = tqdm(total=len(dataset), leave=False)
    while training:
        # Train, compute loss, and update model
        avg_loss += loss.item()
        batch_idx += batch_size
        pbar.update(batch_size)

    avg_loss /= len(dataset) // batch_size
    return avg_loss

@torch.no_grad()
def sample(num_samples=2000):
    mlp.eval()
    z = None # start with noise

    for i in np.arange(num_steps-1, -1, -1):
        eps = None # compute the noise
        
        t = None # get the time step
        z_ = torch.cat([z, t], dim=1)
        z = None # compute z for the next time using your denoiser, eps and z_
    
    z = z.cpu().numpy()
    nll = dataset.calc_nll(z)

    return nll, z

for e in trange(num_epochs):
    train_loss_list.append(train_one_epoch())
    nll, z = sample()
    nll_list.append(nll)
    # dataset.show(z, os.path.join(plot_dir, f"epoch_{e+1}.png"))
    dataset.show(z, os.path.join(plot_dir, f"latest.png"))
    nll_list.append(0)
    print(f"Epoch {e+1}/{num_epochs}, Loss: {train_loss_list[-1]:.4f}")
    scheduler.step()
    if (e + 1) % 1000 == 0:
        dataset.mix_data()

# if you don't have enough GPU space to generate 5000 samples at once,
# you can do it in batches of 100 or whatever size works for you.
nll, z = sample(5000)
dataset.show(z, os.path.join(plot_dir, "final.png"))
np.save(os.path.join(plot_dir, "uncond_gen_samples.pt"), z)
torch.save(mlp.state_dict(), "checkpoints/denoiser.pt")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(train_loss_list)
axs[0].set_title("Training Loss")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].set_yscale("log")

axs[1].plot(nll_list)
axs[1].set_title("Negative Log Likelihood")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("NLL")
axs[1].set_yscale("log")

fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "train_logs.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)
