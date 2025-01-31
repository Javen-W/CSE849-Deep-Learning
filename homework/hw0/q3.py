import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from data import Q3Dataset


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.layer1(x)
        # x = torch.relu(x)
        x = self.layer2(x)
        # x = torch.relu(x)
        x = self.layer3(x)
        return x


"""
Load the dataset from zip files
"""
data = torch.load("HW0_data.pt", weights_only=True)
x_train, y_train = data['x_train'], data['y_train']
x_val, y_val = data['x_val'], data['y_val']
x_test = data['x_test']

"""
Create corresponding datasets.
"""
train_dataset = Q3Dataset(x_train, y_train)
val_dataset = Q3Dataset(x_val, y_val)
test_dataset = TensorDataset(x_test)

"""
Set the seed to the last five digits of your student number.
E.g., if you are student number 160474145, set the seed to 74145.
"""
overall_seed = 59989

"""
Set the seed to the last three digits of your student number.
E.g., if you are student number 160474145, set the seed to 145.
"""
model_seed = 989


def train_model(_seed, batch_size, lr):
    """
    Initializes, trains, and plots a model for the given parameters.
    """
    # global seed
    torch.manual_seed(_seed)

    """
    Create dataloaders for each dataset.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    """
    Create the MLP as described in the PDF
    """
    model = MLP()

    """
    Initialize the model after setting PyTorch seed to model_seed. But we
    also don't want to disturb other parts of the code. Therefore, we will
    store the current random state, set the seed to model_seed, initialize
    the model and restore the random state.
    """
    rng = torch.get_rng_state()
    # TODO: set seed
    torch.manual_seed(model_seed)

    # TODO: Sample the values from a normal distribution
    for param in model.parameters():
        nn.init.normal_(param)
    torch.set_rng_state(rng)

    """
    Create an AdamW optimizer with the given learning rate lr and
    weight_decay. Check the usage of the optimizer in the PyTorch documentation.
    Make sure to pass the parameters of model to the optimizer.
    """
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    """
    Set up the loss function as nn.MSELoss.
    """
    loss_fn = nn.MSELoss()

    """
    Train the model.
    """
    num_epochs = 1000
    for _ in tqdm(range(num_epochs)):
        # TODO: Set your model to training mode.
        model.train()
        for batch in train_loader:
            # TODO: Zero the gradients
            optimizer.zero_grad()
            
            # TODO: Unpack the batch. It is a tuple containing x and y.
            x, y = batch

            # TODO: Pass it through the model to get the predicted y_hat.
            y_hat = model(x)

            # TODO: Calculate the loss using the loss function.
            loss = loss_fn(y_hat, y)

            # TODO: Backpropagate the loss.
            loss.backward()

            # TODO: Update the model weights.
            optimizer.step()
        
    # TODO: Evaluate your model on the validation set.
    # Remember to set the model in evaluation mode and to use torch.no_grad()
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            val_loss += loss.item() * x.size(0)
    val_loss /= len(val_dataset)

    """
    Visualize the model predictions.
    """
    with torch.no_grad():
        predictions = model(x_train.reshape((-1, 1)))
        sorted_indices = torch.argsort(x_train)
        ax.plot(x_train[sorted_indices], predictions[sorted_indices], label=f'Seed {seed}', marker='.', linestyle=':')

    """
    Return model and validation error.
    """
    return model, val_loss

"""
TODO: For each seed, plot the fitted model along with the training
data points. The data samples need to be plotted only once. Follow the
instructions from q2.py on how to plot the results.
"""
seeds_list = [1, 2, 3, 4, 5]
fig, ax = plt.subplots()
for seed in seeds_list:
    train_model(seed, batch_size=2000, lr=1e-2)

# Complete the model prediction visualization.
ax.scatter(x_train, y_train, label='Training Data', c="blue", marker=".")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
ax.set_title('Data and Fitted Models')
ax.legend()
fig.tight_layout()
fig.savefig('./results/q3_plot.png', dpi=300)
plt.clf()
plt.close(fig)

"""
Tune the model hyperparameters (batch_size, learning_rate).
"""
best_loss = float('inf')
best_model = None
batch_list = [16, 32, 64, 2000]
lr_list = [0.001, 0.01, 0.1, 1e-2]
for batch_size in batch_list:
    for lr in lr_list:
        model, val_loss = train_model(
            _seed=overall_seed,
            batch_size=batch_size,
            lr=lr,
        )
        if val_loss < best_loss:
            best_model = model
            best_loss = val_loss
            print(f"Best params: loss={val_loss}, batch_size={batch_size}, lr={lr}")


# TODO: Run the model on the test set
with torch.no_grad():
    yhat_test = best_model(x_test.reshape((-1, 1))).numpy()

with open('./results/q3_test_output.txt', "w") as f:
    for yhat in yhat_test:
        f.write(f"{yhat.item()}\n")

# TODO: Save the model as q3_model.pt
torch.save(best_model.state_dict(), './results/q3_model.pt')
