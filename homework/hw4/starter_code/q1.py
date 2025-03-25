from tqdm import trange, tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from yelp_dataset import YelpDataset
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
emb_dim = 50
batch_size = None
rnn_dropout = None
num_rnn_layers = 2
lr = None
num_epochs = None

train_dataset = YelpDataset("train")
val_dataset = YelpDataset("val")
test_dataset = YelpDataset("test")

# TODO: Load the modified GloVe embeddings to nn.Embedding instance. Set
# freeze=False.
emb_init_tensor = None
embeddings = None
embeddings = embeddings.to(device)

def collate_fn(batch):
    # TODO: Implement a collate_fn function. The function should pack
    # the input and return the stars along with it.

    return input_padded, stars

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, collate_fn=collate_fn)

# TODO: Create the RNN model
model = None
model = model.to(device)

# TODO: Create the linear classifier
classifier = None
classifier = classifier.to(device)

# TODO: Get all parameters and create an optimizer to update them
params = None
optimizer = None

# TODO: Create the loss function
criterion = nn.CrossEntropyLoss()

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

def train_one_epoch():
    avg_loss = 0
    num_steps = 0
    correct = 0
    total_samples = 0
    
    model.train()
    for review, stars in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}"):
        """
        TODO:
        1. Get pass the review through the model to get the output.
        2. Unpack the output and pass the output from the last
        non-padded time-step through the classifier.
        3. Calculate the loss using the criterion.
        4. Update the model parameters.
        """

        with torch.no_grad():
            avg_loss += loss.item()
            total_samples += stars.size(0)
            correct += (torch.argmax(preds, dim=1) == stars-1).sum().item()
            num_steps += 1

    avg_loss /= num_steps
    accuracy = 100*correct/total_samples

    return avg_loss, accuracy

@torch.no_grad()
def validate():
    avg_loss = 0
    num_steps = 0
    correct = 0
    total_samples = 0
    model.eval()
    confusion_matrix = torch.zeros(5, 5)

    for review, stars in tqdm(val_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # TODO: Implement the validation loop similar to the training loop

        with torch.no_grad():
            avg_loss += loss.item()
            total_samples += stars.size(0)
            correct += (torch.argmax(preds, dim=1) == stars-1).sum().item()
            for i in range(stars.size(0)):
                confusion_matrix[stars[i]-1, torch.argmax(preds[i])] += 1
            num_steps += 1

    avg_loss /= num_steps
    accuracy = 100*correct/total_samples

    return avg_loss, accuracy, confusion_matrix

pbar = trange(num_epochs)
for epoch in pbar:
    train_loss, train_accuracy = train_one_epoch()
    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)
    val_loss, val_accuracy, confusion_matrix = validate()
    val_loss_list.append(val_loss)
    val_acc_list.append(val_accuracy)

    pbar.set_postfix({"Train Loss": f"{train_loss:1.3f}", "Train Accuracy": f"{train_accuracy:1.2f}",
                      "Val Loss": f"{val_loss:1.3f}", "Val Accuracy": f"{val_accuracy:1.2f}"})

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(train_loss_list, label="Train")
    axs[0].plot(val_loss_list, label="Val")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(train_acc_list, label="Train")
    axs[1].plot(val_acc_list, label="Val")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_title("Accuracy")
    axs[1].legend()

    fig.tight_layout()
    fig.savefig("plots/q1_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("plots/q1_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

torch.save(model.state_dict(), "q1_model.pt")
torch.save(classifier.state_dict(), "q1_classifier.pt")
torch.save(embeddings.state_dict(), "q1_embedding.pt")
