import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from torchvision.transforms import v2
import sys


torch.manual_seed(123)

from data import create_dataloaders
from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get your dataloaders
train_loader, val_loader = create_dataloaders()

# Get your model
model = CNN()
model.to(device)

# TODO: Initialize weights. You may use kaiming_normal_ for
# initialization. Check this StackOverflow answer:
# https://stackoverflow.com/a/49433937/6211109

# Set your training parameters here
num_epochs = None
lr = None
weight_decay = None

# Setup your cross-entropy loss function
loss_fn = None

# Setup your optimizer that uses lr and weight_decay
optimizer = None

# Setup your learning rate scheduler
scheduler = None

# For plotting.
step = 0
train_step_list = []
train_loss_list = []
train_accuracy_list = []
val_step_list = []
val_loss_list = []
val_accuracy_list = []

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # TODO: Move images and labels to device
        # TODO: Zero the gradients
        # TODO: Forward pass through the model
        # TODO: Calculate the loss
        loss = None

        # TODO: Backward pass
        # TODO: Update weights

        train_loss_list.append(loss.item())
        train_step_list.append(step)

        step += 1
        
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    scheduler.step()
    
    model.eval()
    with torch.no_grad():
        # Compute validation loss and accuracy
        correct, total = 0, 0
        avg_loss = 0.
        for images, labels in val_loader:
            # TODO: Forward pass similar to training
            outputs = None
            loss = None

            avg_loss += loss.item() * labels.size(0)
            # TODO: Get the predicted labels from the model's outputs
            predicted = None
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / total * 100
        avg_loss /= total

        # Similarly compute training accuracy. This training accuracy is
        # not fully reliable as the image transformations are different
        # from the validation transformations. But it will inform you of
        # potential issues.
        train_accuracy = None

        val_loss_list.append(avg_loss)
        val_accuracy_list.append(val_accuracy)
        train_accuracy_list.append(train_accuracy)
        val_step_list.append(step)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Val acc: {val_accuracy:.2f}%,",
              f"Train acc: {train_accuracy:.2f}%")
        
        # Optionally, you can save only your best model so far by
        # keeping track of best validation accuracies.
        torch.save(model.state_dict(), "q1_model.pt")

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(train_step_list, train_loss_list, label="Train")
        axs[0].plot(val_step_list, val_loss_list, label="Val")
        axs[0].set_yscale("log")

        axs[1].plot(val_step_list, train_accuracy_list, label="Train")
        axs[1].plot(val_step_list, val_accuracy_list, label="Val")

        axs[0].set_title("Loss")
        axs[1].set_title("Accuracy")

        for ax in axs:
            ax.legend()
            ax.grid()
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")

        plt.tight_layout()
        plt.savefig(f"q1_plots.png", dpi=300)
        plt.clf()
        plt.close()

torch.save(model.state_dict(), "q1_model.pt")

# You can copy-paste the following code to another program to evaluate
# your model separately.
model.load_state_dict(torch.load("q1_model.pt", weights_only=True))
model.eval()
test_images = sorted(glob("custom_image_dataset/test_unlabeled/*.png"))

# TODO: Create test-time image transformations. Same as what you used
# for validation.
test_tf = None

test_write = open("q1_test.txt", "w")
# We will run through each image and write the predictions to a file.
# You may also write a custom Dataset class to load it more efficiently.
for imgfile in test_images:
    filename = os.path.basename(imgfile)
    img = Image.open(imgfile)
    img = test_tf(img)
    img = img.unsqueeze(0).to(device)
    # TODO: Forward pass through the model and get the predicted label
    predicted = None
    # predicted is a PyTorch tensor containing the predicted label as a
    # single value between 0 and 9 (inclusive)
    test_write.write(f"{filename},{predicted.item()}\n")
test_write.close()
