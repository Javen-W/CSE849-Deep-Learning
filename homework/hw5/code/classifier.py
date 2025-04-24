import os
import torch
from torch import nn
from models import MLP
from data import States
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plot_dir = "outputs/plots/classification"
os.makedirs(plot_dir, exist_ok=True)
checkpoints_dir = "checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)

# Seed
torch.manual_seed(777)

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_steps = 500
batch_size = 10_000
n_epochs = 50
lr = 0.001
weight_decay = 1e-4
n_workers = 0
refresh_interval = 1000  # Refresh noise

# TODO: Create the dataset and the dataloader. Remember to use the same
# number of steps in the dataset as the generation code.
dataset = States(num_steps=n_steps)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, collate_fn=None)

# TODO: create the architecture with the hidden size layers from the PDF.
classifier = MLP(input_dim=3, output_dim=5, hidden_layers=[100, 200, 500]).to(device)

# TODO: Create loss function, optimizer, and scheduler. 
ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
)

label_to_states = {0: "Michigan",
                   1: "Idaho",
                   2: "Ohio",
                   3: "Oklahoma",
                   4: "Wisconsin"}
colors = ["red", "blue", "green", "orange", "purple"]
cmap = ListedColormap(colors)

def train_one_epoch(epoch):
    classifier.train()
    total_loss = 0
    for batch in tqdm(train_loader, leave=False, desc=f"Train epoch {epoch + 1}/{n_epochs}"):
        x_, t, eps, x, y = batch

        optimizer.zero_grad()  # Zero the gradient
        input_ = torch.cat([x_, t], dim=1)  # Concatenate x_t and t
        logits = classifier(input_)  # Forward-feed
        loss = ce_loss(logits, y)  # Calculate loss

        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)  # Clip gradients
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_.size(0)

    avg_loss = total_loss / len(dataset)
    return avg_loss

#TODO: Train the classifier and save it.
train_loss_list = []
for e in trange(n_epochs):
    train_loss = train_one_epoch(e)
    train_loss_list.append(train_loss)
    print(f"Epoch {e+1}/{n_epochs}, Loss: {train_loss:.4f}")
    scheduler.step(train_loss)
    if (e + 1) % refresh_interval == 0:
        dataset.mix_data()
torch.save(classifier.state_dict(), os.path.join(checkpoints_dir, "classifier.pt"))

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
plt.savefig(os.path.join(plot_dir, "classifier_predictions.png"))

