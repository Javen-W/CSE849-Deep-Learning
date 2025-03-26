from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from yelp_dataset import YelpDataset
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skip_training = True

# Parameters
emb_dim = 50
hidden_dim = 50
batch_size = 4
rnn_dropout = 0.35
num_rnn_layers = 2
lr = 0.0001
num_epochs = 5

train_dataset = YelpDataset("train")
val_dataset = YelpDataset("val")
test_dataset = YelpDataset("test")

# TODO: Load the modified GloVe embeddings to nn.Embedding instance. Set freeze=False.
emb_init_tensor = torch.load('code/glove/modified_glove_50d.pt')
if isinstance(emb_init_tensor, dict):
    emb_init_tensor = torch.squeeze(torch.stack(list(emb_init_tensor.values())), 1)
    print(emb_init_tensor.shape)
embeddings = nn.Embedding.from_pretrained(
    emb_init_tensor,
    freeze=False
)
embeddings = embeddings.to(device)

def collate_fn(batch):
    # TODO: Implement a collate_fn function. The function should pack the input and return the stars along with it.
    if isinstance(batch[0], tuple):
        sequences, stars = zip(*batch)
        stars = torch.tensor(stars, dtype=torch.long)
    else:
        sequences = batch
        stars = None

    # Pad sequences
    input_padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    ).to(device)

    # Embed padded sequences
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long).cpu()
    embedded = embeddings(input_padded)

    # Pack padded embeddings
    packed_embeddings = pack_padded_sequence(
        embedded,
        lengths,
        batch_first=True,
        enforce_sorted=False
    )

    # Return
    if stars is not None:
        stars = stars.to(device)
    return packed_embeddings.to(device), stars

# DataLoader setup
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

class RNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        """
        # RNN with 2 layers
        # input_size = embedding_dim (50), hidden_size = 50, num_layers = 2
        """
        super(RNNModel, self).__init__()
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, packed_embeddings):
        """
        packed_embeddings: PackedSequence of shape [batch_size, seq_len, embedding_dim]
        hidden shape: [num_layers, batch_size, hidden_dim]
        """
        packed_output, hidden = self.rnn(packed_embeddings)
        # Take the hidden state from the last layer -> Shape: [batch_size, hidden_dim]
        last_hidden = hidden[-1]
        return self.dropout(last_hidden)


# TODO: Create the RNN model
model = RNNModel(
    embedding_dim=emb_dim,
    hidden_dim=hidden_dim,
    n_layers=num_rnn_layers,
    dropout=rnn_dropout,
)
model = model.to(device)

# TODO: Create the linear classifier
classifier = nn.Linear(
    in_features=hidden_dim,
    out_features=5,  # ratings 0-4
)
classifier = classifier.to(device)

# TODO: Get all parameters and create an optimizer to update them
params = list(embeddings.parameters()) + list(model.parameters()) + list(classifier.parameters())
optimizer = torch.optim.Adam(
    params,
    lr=lr,
    weight_decay=1e-4,
)

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
    classifier.train()
    embeddings.train()

    for reviews, stars in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}"):
        """
        TODO:
        1. Get pass the review through the model to get the output.
        2. Unpack the output and pass the output from the last non-padded time-step through the classifier.
        3. Calculate the loss using the criterion.
        4. Update the model parameters.
        """
        # Pass the review (packed embeddings) through the model
        rnn_output = model(reviews)  # Shape: [batch_size, hidden_dim]

        # Pass the RNN output through the linear classifier
        preds = classifier(rnn_output)  # Shape: [batch_size, 5]

        # Calculate the loss
        loss = criterion(preds, stars)

        # Update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            avg_loss += loss.item()
            total_samples += stars.size(0)
            correct += (torch.argmax(preds, dim=1) == stars).sum().item()
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
    confusion_matrix = torch.zeros(5, 5)

    model.eval()
    classifier.eval()
    embeddings.eval()

    for reviews, stars in tqdm(val_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # TODO: Implement the validation loop similar to the training loop
        # Pass the review (packed embeddings) through the model
        rnn_output = model(reviews)  # Shape: [batch_size, hidden_dim]

        # Pass the RNN output through the linear classifier
        preds = classifier(rnn_output)  # Shape: [batch_size, 5]

        # Calculate the loss
        loss = criterion(preds, stars)

        # Compute metrics
        with torch.no_grad():
            avg_loss += loss.item()
            total_samples += stars.size(0)
            correct += (torch.argmax(preds, dim=1) == stars).sum().item()
            # Update confusion matrix
            for i in range(stars.size(0)):
                true_label = stars[i]
                pred_label = torch.argmax(preds[i])
                confusion_matrix[true_label, pred_label] += 1
            num_steps += 1

    avg_loss /= num_steps
    accuracy = 100*correct/total_samples

    return avg_loss, accuracy, confusion_matrix

if not skip_training:
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
        fig.savefig("results/plots/q1_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

        sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("results/plots/q1_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

    torch.save(model.state_dict(), "results/q1_model.pt")
    torch.save(classifier.state_dict(), "results/q1_classifier.pt")
    torch.save(embeddings.state_dict(), "results/q1_embedding.pt")


@torch.no_grad()
def predict_test_set():
    predictions = []
    for review, _ in test_loader:
        # Forward pass
        rnn_output = model(review)  # Shape: [batch_size, hidden_dim]
        logits = classifier(rnn_output)  # Shape: [batch_size, 5]

        # Get predicted ratings (0 to 4)
        preds = torch.argmax(logits, dim=1)  # Shape: [batch_size]

        # Shift predictions from 0-4 to 1-5
        preds = preds + 1

        # Collect predictions
        predictions.extend(preds.cpu().numpy())

    return predictions

# Load the best checkpoints
model.load_state_dict(torch.load("results/q1_model.pt"))
classifier.load_state_dict(torch.load("results/q1_classifier.pt"))
embeddings.load_state_dict(torch.load("results/q1_embedding.pt"))

# Set models to evaluation mode
embeddings.eval()
model.eval()
classifier.eval()

# Generate predictions
test_predictions = predict_test_set()

# Save predictions to q1_test.txt
with open('results/q1_test.txt', 'w') as f:
    for pred in test_predictions:
        f.write(f"{pred}\n")