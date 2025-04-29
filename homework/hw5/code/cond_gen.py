import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")
from tqdm import trange, tqdm
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
n_steps = 500

# create the same classifier as in classifier.py and load the weights.
# Set it to eval mode.
classifier = MLP(input_dim=3, output_dim=5, hidden_layers=[100, 200, 500]).to(device)
classifier.load_state_dict(torch.load(os.path.join(checkpoints_dir, "classifier.pt"), weights_only=True))
classifier.eval()
logsoftmax = nn.LogSoftmax(dim=0) # create log-softmax

# create your denoiser model architecture and load the weights from uncond_gen.py
denoiser = MLP(input_dim=3, output_dim=2, hidden_layers=[256, 256, 256, 256]).to(device)
denoiser.load_state_dict(torch.load(os.path.join(checkpoints_dir, "denoiser.pt"), weights_only=True))

dataset = States(num_steps=n_steps)
dataset.show(save_to=os.path.join(plot_dir, "original_data.png"))

def sample(label, num_samples=1000):
    denoiser.eval()
    z = torch.randn(num_samples, 2).to(device) # start with random noise

    for i in np.arange(n_steps - 1, 0, -1):
        t = dataset.steps[i].expand(num_samples, 1).to(device) # get the time step
        z_ = torch.cat([z, t], dim=1)
        eps = denoiser(z_) # get the denoiser prediction

        # compute the gradient of the log-softmax classifier w.r.t. the input.
        cls_grad = logsoftmax(classifier(z_))[:, label]
        cls_grad.backward(gradient=torch.ones_like(cls_grad), retain_graph=True)
        cls_grad = cls_grad.reshape(-1, 1)

        # Sampling step
        alpha_bar_t = dataset.alpha_bar[i].to(device)
        alpha_bar_tm1 = dataset.alpha_bar[i - 1].to(device)
        eps_hat = eps - torch.sqrt(1 - alpha_bar_t) * cls_grad
        z = torch.sqrt(alpha_bar_tm1) * ((z - torch.sqrt(1 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)) + torch.sqrt(1 - alpha_bar_tm1) * eps_hat
 
    z = z.detach().cpu().numpy()
    nll = dataset.calc_nll(z)
    return nll, z

for label in range(5):
    full_z = []
    for i in trange(5):
        nll, z = sample(label)
        full_z.append(z)
    full_z = np.concatenate(full_z, axis=0)
    nll = dataset.calc_nll(full_z)
    print(f"Label {label}, NLL: {nll:.4f}")
    dataset.show(z, os.path.join(plot_dir, f"label_{label}.png"))
    np.save(os.path.join(plot_dir, f"cond_gen_samples_{label}.npy"), full_z)

