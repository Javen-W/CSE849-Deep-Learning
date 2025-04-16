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
    for batch in tqdm(train_loader, leave=False, desc=f"Train epoch {epoch + 1}/{n_epochs}"):
        x_, t, eps, x, y = batch
        t = t.reshape(-1, 1)  # Reshape for MLP input

        optimizer.zero_grad() # Zero the gradient
        input_ = torch.cat([x_, t], dim=1) # Concatenate x_t and t
        eps_logits = mlp(input_) # Forward-feed
        loss = mse_loss(eps_logits, eps)  # Calculate loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_.size(0)

    avg_loss = total_loss / len(dataset)
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
