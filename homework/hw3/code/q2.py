import torch
import imageio.v2 as imio
import os

from model import CNN

model = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the weights
model.load_state_dict(torch.load("q1_model.pt", weights_only=True))

model.eval()

conv_weights = None # Get the conv1 layer weights

os.makedirs("q2_filters", exist_ok=True)

for i in range(conv_weights.shape[0]):
    f = None # get the i-th filter
    # TODO: Normalize the filter to [0, 255] as convert it to uint8.
    # Otherwise, it will not be visualized correctly.
    imio.imwrite(f"q2_filters/filter_{i}.png", f)
