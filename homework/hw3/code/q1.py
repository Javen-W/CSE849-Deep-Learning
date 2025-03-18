import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from torchvision.transforms import v2
import sys

torch.manual_seed(123)

from data import create_dataloaders, NORMAL_MEAN, NORMAL_STD, DATASET_ROOT
from model import CNN

skip_training = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get your dataloaders
train_loader, val_loader = create_dataloaders()

# Get your model
model = CNN()
model.to(device)

# TODO: Initialize weights. You may use kaiming_normal_ for
# initialization. Check this StackOverflow answer: https://stackoverflow.com/a/49433937/6211109
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
model.apply(init_weights)

# Set your training parameters here
num_epochs = 50
lr = 0.01
weight_decay = 1e-4

# Setup your cross-entropy loss function
loss_fn = nn.CrossEntropyLoss()

# Setup your optimizer that uses lr and weight_decay
optimizer = torch.optim.Adam(
    lr=lr,
    weight_decay=weight_decay,
    params=model.parameters(),
)

# Setup your learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=num_epochs,
)

# For plotting.
step = 0
train_step_list = []
train_loss_list = []
train_accuracy_list = []
val_step_list = []
val_loss_list = []
val_accuracy_list = []

if not skip_training:
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # TODO: Move images and labels to device
            images, labels = images.to(device), labels.to(device)

            # TODO: Zero the gradients
            optimizer.zero_grad()

            # TODO: Forward pass through the model
            outputs = model(images)

            # TODO: Calculate the loss
            loss = loss_fn(outputs, labels)

            # TODO: Backward pass
            loss.backward()

            # TODO: Update weights
            optimizer.step()

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
            avg_loss = 0.0
            for images, labels in val_loader:
                # TODO: Forward pass similar to training
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                avg_loss += loss.item() * labels.size(0)

                # TODO: Get the predicted labels from the model's outputs
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_accuracy = correct / total * 100
            avg_loss /= total
            val_loss_list.append(avg_loss)
            val_accuracy_list.append(val_accuracy)
            val_step_list.append(step)

            # Similarly compute training accuracy. This training accuracy is
            # not fully reliable as the image transformations are different
            # from the validation transformations. But it will inform you of
            # potential issues.
            correct, total = 0, 0
            for images, labels in train_loader:
                # TODO: Forward pass similar to training
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # TODO: Get the predicted labels from the model's outputs
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = correct / total * 100
            train_accuracy_list.append(train_accuracy)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Val acc: {val_accuracy:.4f}%,",
                  f"Train acc: {train_accuracy:.4f}%")

            # Optionally, you can save only your best model so far by
            # keeping track of best validation accuracies.
            torch.save(model.state_dict(), "results/q1_model.pt")

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
            plt.savefig(f"results/q1_plots.png", dpi=300)
            plt.clf()
            plt.close()

    torch.save(model.state_dict(), "results/q1_model.pt")

# You can copy-paste the following code to another program to evaluate
# your model separately.
model.load_state_dict(torch.load("results/q1_model.pt", weights_only=True))
model.eval()
test_images = sorted(glob(os.path.join(DATASET_ROOT, "test_unlabeled/*.png")))

# TODO: Create test-time image transformations. Same as what you used
# for validation.
test_tf = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=NORMAL_MEAN, std=NORMAL_STD),
])

test_write = open("results/q1_test.txt", "w")
# We will run through each image and write the predictions to a file.
# You may also write a custom Dataset class to load it more efficiently.
for imgfile in test_images:
    filename = os.path.basename(imgfile)
    img = Image.open(imgfile)
    img = test_tf(img)
    img = img.unsqueeze(0).to(device)
    # TODO: Forward pass through the model and get the predicted label
    outputs = model(img)
    predicted = torch.argmax(outputs, dim=-1)
    # predicted is a PyTorch tensor containing the predicted label as a
    # single value between 0 and 9 (inclusive)
    test_write.write(f"{filename},{predicted.item()}\n")
test_write.close()
