import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from data import Q3Dataset
from q2 import model_seed

seeds_list = [1, 2, 3, 4, 5]
# TODO: For each seed, plot the fitted model along with the training
# data points. The data samples need to be plotted only once. Follow the
# instructions from q2.py on how to plot the results.
for seed in seeds_list:
    torch.manual_seed(seed)

    """
    Load the dataset from zip files
    """
    data = torch.load("HW0_data.pt", weights_only=True)

    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    y_val = data['y_val']
    x_test = data['x_test']
    y_test = data['y_test']

    """
    Create corresponding datasets.
    """
    train_dataset = Q3Dataset(x_train, y_train)
    val_dataset = Q3Dataset(x_val, y_val)
    test_dataset = Q3Dataset(x_test, y_test)

    """
    Create dataloaders for each dataset.
    """
    batch_size = 2000 # Use the entire dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False)

    """
    Create the MLP as described in the PDF
    """
    model = nn.Sequential(
        nn.Linear(1, 10),
        # nn.ReLU(),
        nn.Linear(10, 10),
        # nn.ReLU(),
        nn.Linear(10, 1),
        # nn.Sigmoid()
    )

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
    nn.init.normal_(model.weight)
    torch.set_rng_state(rng)

    """
    Create an AdamW optimizer with the given learning rate lr and
    weight_decay. Check the usage of the optimizer in the PyTorch documentation.
    Make sure to pass the parameters of model to the optimizer.
    """
    lr = 1e-2
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    """
    Set up the loss function as nn.MSELoss.
    """
    loss_fn = None

    num_epochs = 1000

    train_step_list = []
    train_loss_list = []
    val_step_list = []
    val_loss_list = []
    w_list = []
    step = 0

    for e in trange(num_epochs):
        # TODO: Set your model to training mode.

        for batch in tqdm(train_loader, leave=False, desc="Training"):
            continue
            # TODO: Zero the gradients
            
            # TODO: Unpack the batch. It is a tuple containing x and y.

            # TODO: Pass it through the model to get the predicted y_hat.

            # TODO: Calculate the loss using the loss function.

            # TODO: Backpropagate the loss.

            # TODO: Update the model weights.
        
            # TODO: Store the training loss in the list
        
        # TODO: Evaluate your model on the validation set.
        # Remember to set the model in evaluation mode and to use
        # torch.no_grad()

# TODO: Run the model on the test set
with torch.no_grad():
    yhat_test = None

with open("q3_test_output.txt", "w") as f:
    for yhat in yhat_test:
        f.write(f"{yhat.item()}\n")

# TODO: Save the model as q3_model.pt
