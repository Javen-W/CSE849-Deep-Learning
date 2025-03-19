import numpy as np
import torch
import imageio.v2 as imio
import os

from model import CNN

model = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the weights
model.load_state_dict(torch.load("results/q1_model.pt", weights_only=True))

model.eval()

conv_weights = model.conv1.weight # Get the conv1 layer weights

os.makedirs("q2_filters", exist_ok=True)

for i in range(conv_weights.shape[0]):
    # Get the i-th filter - shape will be [3, 7, 7]
    f = conv_weights[i].detach().cpu().numpy()

    # Take mean across channels to get 7x7 filter
    f = np.mean(f, axis=0)

    # Normalize to [0, 255]
    f_min, f_max = f.min(), f.max()
    if f_max > f_min:  # Avoid division by zero
        f = (f - f_min) / (f_max - f_min)  # Scale to [0, 1]
        f = (f * 255).astype(np.uint8)  # Convert to uint8
    else:
        f = np.zeros_like(f, dtype=np.uint8)  # If filter is constant, make it zero

    # Save using imageio
    imio.imwrite(f"q2_filters/filter_{i}.png", f)
