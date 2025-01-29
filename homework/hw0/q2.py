import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from data import MyDataset

"""
Set the seed to the last five digits of your student number.
E.g., if you are student number 160474145, set the seed to 74145.
"""
overall_seed = None

"""
Set the seed to the last three digits of your student number.
E.g., if you are student number 160474145, set the seed to 145.
"""
model_seed = None

# TODO: Set the seed

"""
Generate the dataset as describe in the HW pdf file. Complete the class
MyDataset in data.py.

NOTE: Do not change the values of alpha, beta, and num_samples.
"""

alpha = 2
beta = 0.5
num_samples = 2000

train_dataset = MyDataset(alpha, beta, num_samples, 'train', overall_seed)
val_dataset = MyDataset(alpha, beta, num_samples, 'val', overall_seed)

"""
Create a dataloader for each dataset.

NOTE: Do not change the values for batch_size. The number of workers
can be adjusted according to your system.

NOTE: Set shuffle to True for the training dataset and False for the
validation dataset. Set drop_last to True for the training dataset and
False for the validation dataset.
"""

batch_size = num_samples # Use the entire dataset
train_loader = None
val_loader = None

"""
TODO: Create a simple linear regression model using nn.Linear.
"""
model = None

"""
TODO: Initialize the model after setting PyTorch seed to model_seed. But we
also don't want to disturb other parts of the code. Therefore, we will
store the current random state, set the seed to model_seed, initialize
the model and restore the random state.
"""
rng = torch.get_rng_state()
# TODO: set seed here

# TODO: Sample the values from a normal distribution
torch.set_rng_state(rng)

"""
TODO: Create an AdamW optimizer with the given learning rate lr and
weight_decay. Check the usage of the optimizer in the PyTorch documentation.
Make sure to pass the parameters of model to the optimizer.

NOTE: Do not change the values of lr and weight_decay.
"""
lr = 1e-2
weight_decay = 1e-4
optimizer = None

"""
TODO: Set up the loss function as nn.MSELoss.
"""
loss_fn = None

"""
NOTE: Do not change the below values for num_epochs
"""
num_epochs = 1000

train_step_list = []
train_loss_list = []
val_step_list = []
val_loss_list = []
w_list = []
step = 0

for e in trange(num_epochs):
    # TODO: Set your model to training mode.
    model.train()

    for batch in tqdm(train_loader, leave=False, desc="Training"):
        # TODO: Zero the gradients
        
        # TODO: Unpack the batch. It is a tuple containing x and y.

        # TODO: Pass it through the model to get the predicted y_hat.

        # TODO: Calculate the loss using the loss function.
        loss = None

        # TODO: Backpropagate the loss (use the backward function as instructed. There is no need to implement you own function for this project.)

        # TODO: Update the model weights.
    
        # TODO: Store the training loss in the list
        train_step_list.append(step)
        train_loss_list.append(loss.item())
        step += 1
    
    # Evaluate your model on the validation set

    # TODO: Set the model to eval and use torch.no_grad()

print("True parameters: alpha =", alpha, "beta =", beta)
print("Estimated parameters: alpha =", model.weight.item(), "beta =", model.bias.item())
print("Prediction error on validation set:", loss.item())

# Plot the training and validation loss
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(train_step_list, train_loss_list, label='Train Loss')
axs[0].plot(val_step_list, val_loss_list, label='Validation Loss')
axs[0].set_xlabel('Step')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].set_title('Loss vs Step')
axs[0].set_yscale('log')
axs[0].legend()

val_x = val_dataset.x
val_y = val_dataset.y

x_min = val_x.min() - 0.2 * val_x.min().abs()
x_max = val_x.max() + 0.2 * val_x.max().abs()
x_test = torch.linspace(x_min, x_max, 1000).view(-1, 1)

model.eval()
with torch.no_grad():
    y_hat = model(x_test).numpy()

axs[1].scatter(val_x, val_y, label='Data', c="blue", marker=".")
axs[1].plot(x_test, y_hat, label='Fitted Model', c="red")
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].grid(True)
axs[1].set_title('Data and Fitted Model')
axs[1].legend()

fig.tight_layout()
fig.savefig('q2_plot.png', dpi=300)
plt.clf()
plt.close(fig)

# TODO: Save the model as q2_model.pt
