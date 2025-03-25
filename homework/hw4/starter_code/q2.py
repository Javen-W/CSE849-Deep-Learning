from tqdm import trange, tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from positional_encoding import PositionalEncoding
from pig_latin_sentences import PigLatinSentences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
num_tokens = 30
emb_dim = 100
batch_size = None
lr = None
num_epochs = None

# Character to integer mapping
alphabets = "abcdefghijklmnopqrstuvwxyz"
char_to_idx = {}
idx = 0
for char in alphabets:
    char_to_idx[char] = idx
    idx += 1
char_to_idx[' '] = idx
char_to_idx['<sos>'] = idx + 1
char_to_idx['<eos>'] = idx + 2
char_to_idx['<pad>'] = idx + 3

# reverse, integer to character mapping
idx_to_char = {}
for char, idx in char_to_idx.items():
    idx_to_char[idx] = char

@torch.no_grad()
def decode_output(output_logits, expected_words, idx_to_char):
    out_words = output_logits.argmax(2).detach().cpu().numpy()
    expected_words = expected_words.detach().cpu().numpy()
    out_decoded = []
    exp_decoded = []
    pad_pos = char_to_idx['<pad>']
    for i in range(output_logits.size(1)):
        out_decoded.append("".join([idx_to_char[idx] for idx in out_words[:, i] if idx != pad_pos]))
        exp_decoded.append("".join([idx_to_char[idx] for idx in expected_words[:, i] if idx != pad_pos]))

    return out_decoded, exp_decoded

train_dataset = PigLatinSentences("train", char_to_idx)
val_dataset = PigLatinSentences("val", char_to_idx)
test_dataset = PigLatinSentences("test", char_to_idx)

# TODO: Define your embedding
embedding = None
embedding = embedding.to(device)

# TODO: Write your collate_fn
def collate_fn(batch):
    """
    input_sequence is a sequence of embeddings corresponding to the
    English sentence
    output_sequence is a sequence of embeddings corresponding to the
    Pig Latin sentence
    output_padded is the output_sequence padded to the maximum sequence
    length in the batch. This is raw text, not embeddings.
    """
    return input_sequence, output_sequence, output_padded

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, collate_fn=collate_fn)

# TODO: Create your Transformer model
model = None
model = model.to(device)

# TODO: Create your decoder from embedding space to the vocabulary space
decoder = None
decoder = decoder.to(device)

# Your positional encoder
pos_enc = PositionalEncoding(emb_dim)

# TODO: Get all parameters to optimize and create your optimizer
params = None
optimizer = None

# Set up your loss functions
mse_criterion = None
ce_criterion = None

# Store your intermediate results for plotting
epoch_list = []
train_mse_loss_list = []
train_ce_loss_list = []
train_acc_list = []
val_mse_loss_list = []
val_ce_loss_list = []
val_acc_list = []

def compare_outputs(output_text, expected_text):
    correct = 0
    for i in range(len(output_text)):
        out = output_text[i]
        exp = expected_text[i]
        exp = exp.split("<sos>")[1] # remove <sos>
        # remove <eos>
        if "<eos>" in out:
            out = out.split("<eos>")[0]
        exp = exp.split("<eos>")[0]

        if out == exp:
            correct += 1
    
    return correct
        

def train_one_epoch():
    avg_mse_loss = 0
    avg_ce_loss = 0
    total = 0
    correct = 0
    num_samples = 0

    model.train()
    for input_emb, target_emb, target_words in tqdm(train_loader, leave=False, desc=f"Train epoch {epoch+1}/{num_epochs}"):
        """
        TODO:
        1. Get the input and target embeddings
        2. Pass them through the positional encodings.
        3. Create the src_mask and tgt_mask.
        4. Pass the input and target embeddings through the model.
        5. Pass the output embeddings through the decoder.
        6. Calculate the MSE loss between the output embeddings and the
        target embeddings. Remember to use the target embeddings without
        the positional encoding.
        7. Calculate the CE loss between the output logits and the target
        words. Remember to reshape the output logits and target words to
        remove the padding tokens.
        8. Add the MSE and CE losses and backpropagate.
        9. Update the parameters.
        """

        avg_mse_loss += mse_loss.item()
        avg_ce_loss += ce_loss.item()
        total += 1

        with torch.no_grad():
            output_text, expected_text = decode_output(output_logits, target_words, idx_to_char)
            correct += compare_outputs(output_text, expected_text)
            num_samples += len(output_text)

    # display the decoded outputs only for the last step of each epoch
    rand_idx = [_.item() for _ in torch.randint(0, len(output_text),
                                                (min(10, len(output_text)),))]
    for i in rand_idx:
        out_ = output_text[i]
        exp_ = expected_text[i]
        print(f"Train Output:   \"{out_}\"")
        print(f"Train Expected: \"{exp_}\"")
        print("----"*40)

    return avg_mse_loss / total, avg_ce_loss / total, correct / num_samples

@torch.no_grad()
def validate():
    avg_mse_loss = 0
    avg_ce_loss = 0
    total = 0
    correct = 0
    num_samples = 0

    model.eval()
    for input_emb, target_emb, target_words in tqdm(val_loader, leave=False, desc=f"Val epoch {epoch+1}/{num_epochs}"):
        """
        TODO:
        1. Similar to the training loop, set up the embeddings for the
        forward pass. But this time, we only pass the <SOS> token in the
        first step.
        2. The decoded output will be stored in seq_out.
        3. In the next time step, we will pass the input embedding and
        the embeddings for seq_out through the model.
        4. seq_out is updated with the newly generated token.
        5. Repeat this until the maximum sequence length is reached.
        """
        seq_out = []
        while len(seq_out) < target_emb.size(0) - 1:
            pass # FILL THIS

        avg_mse_loss += mse_loss.item()
        avg_ce_loss += ce_loss.item()
        total += 1

        with torch.no_grad():
            output_text, expected_text = decode_output(output_logits, target_words, idx_to_char)
            correct += compare_outputs(output_text, expected_text)
            num_samples += len(output_text)

    # display the decoded outputs only for the last step of each epoch
    rand_idx = [_.item() for _ in torch.randint(0, len(output_text),
                                                (min(10, len(output_text)),))]
    for i in rand_idx:
        out_ = output_text[i]
        exp_ = expected_text[i]
        print(f"Val Output:   \"{out_}\"")
        print(f"Val Expected: \"{exp_}\"")
        print("----"*40)

    return avg_mse_loss / total, avg_ce_loss / total, correct / num_samples


for epoch in trange(num_epochs):
    train_mse_loss, train_ce_loss, train_acc = train_one_epoch()
    val_mse_loss, val_ce_loss, val_acc = validate()
    train_mse_loss_list.append(train_mse_loss)
    train_ce_loss_list.append(train_ce_loss)
    train_acc_list.append(train_acc)
    val_mse_loss_list.append(val_mse_loss)
    val_ce_loss_list.append(val_ce_loss)
    val_acc_list.append(val_acc)

train_mse_loss_list = np.array(train_mse_loss_list)
train_ce_loss_list = np.array(train_ce_loss_list)
train_acc_list = np.array(train_acc_list)*100
val_mse_loss_list = np.array(val_mse_loss_list)
val_ce_loss_list = np.array(val_ce_loss_list)
val_acc_list = np.array(val_acc_list)*100

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(np.arange(num_epochs), train_ce_loss_list + train_mse_loss_list, label="Train")
axs[0, 0].plot(np.arange(num_epochs), val_ce_loss_list + val_mse_loss_list, label="Val")
axs[0, 0].legend()
axs[0, 0].set_title("Total Loss")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].set_yscale("log")

axs[0, 1].plot(np.arange(num_epochs), train_acc_list, label="Train")
axs[0, 1].plot(np.arange(num_epochs), val_acc_list, label="Val")
axs[0, 1].legend()
axs[0, 1].set_title("Accuracy")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Accuracy (%)")

axs[1, 0].plot(np.arange(num_epochs), train_mse_loss_list, label="Train")
axs[1, 0].plot(np.arange(num_epochs), val_mse_loss_list, label="Val")
axs[1, 0].legend()
axs[1, 0].set_title("MSE Loss")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Loss")
axs[1, 0].set_yscale("log")

axs[1, 1].plot(np.arange(num_epochs), train_ce_loss_list, label="Train")
axs[1, 1].plot(np.arange(num_epochs), val_ce_loss_list, label="Val")
axs[1, 1].legend()
axs[1, 1].set_title("CE Loss")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Loss")
axs[1, 1].set_yscale("log")

fig.tight_layout()
fig.savefig("plots/q2_results.png", dpi=300)
plt.close()

print("Final accuracy")
print(f"Train: {train_acc_list[-1]:1.2f}")
print(f"Val: {val_acc_list[-1]:1.2f}")
print("Final losses")
print(f"Train MSE: {train_mse_loss_list[-1]:1.3f}")
print(f"Train CE: {train_ce_loss_list[-1]:1.3f}")
print(f"Val MSE: {val_mse_loss_list[-1]:1.3f}")
