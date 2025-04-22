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

import faulthandler
faulthandler.enable()

plot_dir = "outputs/plots/unconditional_generation"
os.makedirs(plot_dir, exist_ok=True)

# Seed
torch.manual_seed(777)

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 10_000
n_epochs = 5_000
lr = 0.001
weight_decay = 1e-4
n_steps = 500
n_workers = 0
refresh_interval = 1000  # Refresh noise every 1000 epochs

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

def custom_collate(batch):
    # batch is a list of tuples (x_, t, eps, x, y) from States.__getitem__
    x_ = torch.stack([item[0] for item in batch]).to(device)  # [batch_size, 2]
    t = torch.stack([item[1] for item in batch]).to(device)  # [batch_size, 1]
    eps = torch.stack([item[2] for item in batch]).to(device)  # [batch_size, 2]
    x = torch.stack([item[3] for item in batch]).to(device)  # [batch_size, 2]
    y = torch.tensor([item[4] for item in batch], dtype=torch.long, device=device)  # [batch_size]
    return x_, t, eps, x, y

# Create the dataset and dataloader
print("Creating dataset...")
dataset = States(num_steps=n_steps)
print("Dataset created")
dataset.show(save_to=os.path.join(plot_dir, "original_data.png"))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, collate_fn=custom_collate)

train_loss_list = []
nll_list = []

def train_one_epoch(epoch):
    mlp.train()
    total_loss = 0
    try:
        for batch in tqdm(train_loader, leave=False, desc=f"Train epoch {epoch + 1}/{n_epochs}"):
            x_, t, eps, x, y = batch
            x_ = x_.to(device)
            t = t.to(device)
            eps = eps.to(device)

            optimizer.zero_grad() # Zero the gradient
            input_ = torch.cat([x_, t], dim=1) # Concatenate x_t and t
            eps_logits = mlp(input_) # Forward-feed
            loss = mse_loss(eps_logits, eps)  # Calculate loss

            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)  # Clip gradients
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_.size(0)
            torch.cuda.empty_cache()  # Free unused memory
            # torch.cuda.synchronize()  # Ensure CUDA ops complete
    except RuntimeError as e:
        print(f"Error in epoch {epoch + 1}: {e}")
        raise

    avg_loss = total_loss / len(dataset)
    return avg_loss

@torch.no_grad()
def sample(num_samples=2000):
    mlp.eval()
    z = torch.randn(num_samples, 2).to(device)  # Start with noise

    for i in range(n_steps - 1, -1, -1):
        t = dataset.steps[i].expand(num_samples, 1)
        z_ = torch.cat([z, t], dim=1)
        eps = mlp(z_)
        alpha_bar_t = dataset.alpha_bar[i]
        alpha_t = dataset.alpha[i]
        beta_t = dataset.beta[i]

        # DDPM sampling step
        z = (z - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_t)
        if i > 0:  # Add noise except at t=0
            z += torch.sqrt(beta_t) * torch.randn_like(z)

    z = z.cpu().numpy()
    nll = dataset.calc_nll(z)
    return nll, z

for e in trange(n_epochs):
    train_loss = train_one_epoch(e)
    train_loss_list.append(train_loss)
    nll, z = sample()
    nll_list.append(nll)
    # dataset.show(z, os.path.join(plot_dir, f"epoch_{e+1}.png"))
    dataset.show(z, os.path.join(plot_dir, f"latest.png"))
    nll_list.append(0)
    print(f"Epoch {e+1}/{n_epochs}, Loss: {train_loss:.4f}")
    scheduler.step(train_loss)
    if (e + 1) % refresh_interval == 0:
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
