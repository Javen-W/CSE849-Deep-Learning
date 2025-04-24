import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")

from models import MLP
from data import States

plot_dir = "outputs/plots/conditional_diffusion"
os.makedirs(plot_dir, exist_ok=True)
checkpoints_dir = "checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)

# Seed
torch.manual_seed(777)

# training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = None
num_epochs = None
n_steps = 500

# create the same classifier as in classifier.py and load the weights.
# Set it to eval mode.
classifier = None
logsoftmax = None # create log-softmax

# create your denoiser model architecture and load the weights from uncond_gen.py
mlp = None

dataset = States(num_steps=n_steps)
dataset.show(save_to=os.path.join(plot_dir, "original_data.png"))

def sample(label, num_samples=1000):
    mlp.eval()
    z = None # start with random noise

    for i in np.arange(n_steps-1, 0, -1):
        t = None # get the time step
        z_ = torch.cat([z, t], dim=1)
        eps = None # get the denoiser prediction
        # compute the gradient of the log-softmax classifier w.r.t. the
        # input.
        cls_grad = None
        eps_hat = None # compute eps_hat
        z = None # compute z for the next step
 
    z = z.detach().cpu().numpy()
    nll = dataset.calc_nll(z)

    return nll, z

for label in range(5):
    full_z = []
    for i in range(5):
        nll, z = sample(label)
        full_z.append(z)
    full_z = np.concatenate(full_z, axis=0)
    nll = dataset.calc_nll(full_z)
    print(f"Label {label}, NLL: {nll:.4f}")
    dataset.show(z, os.path.join(plot_dir, f"label_{label}.png"))
    np.save(os.path.join(plot_dir, f"cond_gen_samples_{label}.npy"), full_z)

