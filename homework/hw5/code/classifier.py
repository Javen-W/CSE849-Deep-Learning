import torch
from torch import nn
from models import MLP
from data import States
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# TODO: Create the dataset and the dataloader. Remember to use the same
# number of steps in the dataset as the generation code.

# TODO: create the architecture with the hidden size layers from the
# PDF.
classifier = None

# TODO: Set the training parameters.
lr = None
num_epochs = None

# TODO: Create loss function, optimizer, and scheduler. 
ce_loss = None
optimizer = None
scheduler = None

label_to_states = {0: "Michigan",
                   1: "Idaho",
                   2: "Ohio",
                   3: "Oklahoma",
                   4: "Wisconsin"}
colors = ["red", "blue", "green", "orange", "purple"]
cmap = ListedColormap(colors)

#TODO: Train the classifier and save it.
torch.save(classifier.state_dict(), "classifier.pt")

clean_X = dataset.data
labels = dataset.labels
classifier.eval()
inp = torch.cat([clean_X, torch.zeros(clean_X.shape[0], 1)], dim=1).to(device)
with torch.no_grad():
    all_preds = classifier(inp).argmax(1).detach().cpu()

clean_X = clean_X.cpu()
labels = labels.cpu()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.set_title("Classifier Predictions")
ax.set_xlabel("X")
ax.set_ylabel("Y")
im = ax.scatter(clean_X[:, 0], clean_X[:, 1], s=1,
           c=all_preds, cmap=cmap)

X_max1 = clean_X.max(0).values[0].item()
X_max2 = clean_X.max(0).values[1].item()
X_min1 = clean_X.min(0).values[0].item()
X_min2 = clean_X.min(0).values[1].item()

X, Y = torch.meshgrid(
    torch.linspace(X_min1, X_max1, 100),
    torch.linspace(X_min2, X_max2, 100)
)
X = X.flatten()
Y = Y.flatten()

with torch.no_grad():
    grid_X = torch.cat([X.unsqueeze(1), Y.unsqueeze(1), torch.zeros(X.size(0), 1)], dim=1)
    grid_X = grid_X.to(device)
    grid_preds = classifier(grid_X)
    grid_preds = grid_preds.argmax(1).detach().cpu()
    grid_preds = grid_preds.reshape(100, 100)

X = X.reshape(100, 100)
Y = Y.reshape(100, 100)

ax.contourf(X, Y, grid_preds, alpha=0.3,
            cmap=cmap, levels=5)
cbar = fig.colorbar(im, label="States")
cbar.set_ticks(np.arange(5)*0.8 + 0.4, labels=list(label_to_states.values()))
plt.savefig("classifier_predictions.png")

