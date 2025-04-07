from torch.nn.utils.rnn import pad_sequence
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from positional_encoding import PositionalEncoding
from pig_latin_sentences import PigLatinSentences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skip_training = False
skip_validation = True

# Parameters
num_tokens = 30
emb_dim = 100
batch_size = 32
lr = 0.0001
num_epochs = 10
num_workers = 0

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
def decode_output(output_logits, expected_words):
    out_words = output_logits.argmax(dim=-1).detach().cpu().numpy()  # (batch_size, seq_len)
    expected_words = expected_words.detach().cpu().numpy()  # (batch_size, seq_len)
    out_decoded = []
    exp_decoded = []
    pad_idx = char_to_idx['<pad>']
    eos_idx = char_to_idx['<eos>']

    for i in range(out_words.shape[0]):  # Iterate over batch
        # Decode output sequence
        out_seq = []
        for idx in out_words[i]:
            if idx == pad_idx:
                continue
            out_seq.append(idx_to_char[idx])
            if idx == eos_idx:
                break
        out_decoded.append("".join(out_seq))

        # Decode expected sequence
        exp_seq = []
        for idx in expected_words[i]:
            if idx == pad_idx:
                continue
            exp_seq.append(idx_to_char[idx])
            if idx == eos_idx:
                break
        exp_decoded.append("".join(exp_seq))

    return out_decoded, exp_decoded

def compare_outputs(output_text, expected_text):
    correct = 0
    for out, exp in zip(output_text, expected_text):
        # Remove <sos> prefix if present
        out = out.replace("<sos>", "")
        exp = exp.replace("<sos>", "")
        # Take content before <eos> if present
        out = out.split("<eos>")[0] if "<eos>" in out else out
        exp = exp.split("<eos>")[0] if "<eos>" in exp else exp
        # Strip padding and whitespace
        out = out.strip()
        exp = exp.strip()
        if out == exp:
            correct += 1
    return correct

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
    eng_batch, pig_batch = zip(*batch)

    # Pad sequences
    eng_padded = pad_sequence(eng_batch, batch_first=True, padding_value=char_to_idx['<pad>']).to(device)
    pig_padded = pad_sequence(pig_batch, batch_first=True, padding_value=char_to_idx['<pad>']).to(device)

    # Embed sequences
    input_sequence = embedding(eng_padded)
    output_sequence = embedding(pig_padded)
    output_padded = pig_padded

    return input_sequence, output_sequence, output_padded

# Create Datasets
train_dataset = PigLatinSentences("train", char_to_idx)
val_dataset = PigLatinSentences("val", char_to_idx)
test_dataset = PigLatinSentences("test", char_to_idx)

# TODO: Define your embedding
embedding = nn.Embedding(
    num_embeddings=num_tokens,
    embedding_dim=emb_dim,
    padding_idx=char_to_idx['<pad>'],
)
embedding = embedding.to(device)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=num_workers)

# TODO: Create your Transformer model
model = nn.Transformer(
    d_model=emb_dim,
    nhead=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    batch_first=True,
)
model = model.to(device)

# TODO: Create your decoder from embedding space to the vocabulary space
decoder = nn.Linear(
    in_features=emb_dim,
    out_features=num_tokens,
)
nn.init.xavier_uniform_(decoder.weight)
nn.init.zeros_(decoder.bias)
decoder = decoder.to(device)

# Your positional encoder
pos_enc = PositionalEncoding(emb_dim)

# TODO: Get all parameters to optimize and create your optimizer
params = list(embedding.parameters()) + list(model.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(
    params,
    lr=lr,
    weight_decay=1e-4,
)

# Set up your loss functions
mse_criterion = nn.MSELoss()
ce_criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<pad>'])

# Store your intermediate results for plotting
epoch_list = []
train_mse_loss_list = []
train_ce_loss_list = []
train_acc_list = []
val_mse_loss_list = []
val_ce_loss_list = []
val_acc_list = []

def train_one_epoch(epoch):
    avg_mse_loss = 0
    avg_ce_loss = 0
    total_batches = 0
    total_correct = 0
    total_samples = 0

    model.train()
    embedding.train()
    decoder.train()

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
        # Zero the gradient
        optimizer.zero_grad()

        # Add positional encoding
        input_pos = pos_enc(input_emb)
        target_pos = pos_enc(target_emb)

        # Create masks
        src_mask = model.generate_square_subsequent_mask(input_emb.size(1)).to(device)
        tgt_mask = model.generate_square_subsequent_mask(target_emb.size(1)).to(device)

        # Forward pass
        output_emb = model(
            src=input_pos,
            tgt=target_pos,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_is_causal=True,
            tgt_is_causal=True
        )

        # Decode to vocabulary space
        output_logits = decoder(output_emb)

        # Calculate the losses
        mse_loss = mse_criterion(output_emb, target_emb)
        ce_loss = ce_criterion(
            output_logits.view(-1, num_tokens),
            target_words.view(-1)
        )

        # Update the model parameters
        total_loss = mse_loss + ce_loss
        total_loss.backward()
        optimizer.step()

        # Update metrics
        avg_mse_loss += mse_loss.item()
        avg_ce_loss += ce_loss.item()
        total_batches += 1

        with torch.no_grad():
            output_text, expected_text = decode_output(output_logits, target_words)
            total_correct += compare_outputs(output_text, expected_text)
            total_samples += len(output_text)

    # Display the decoded outputs only for the last step of each epoch
    rand_idx = [_.item() for _ in torch.randint(0, len(output_text), (min(10, len(output_text)),))]
    for i in rand_idx:
        out_ = output_text[i]
        exp_ = expected_text[i]
        print(f"Train Output:   \"{out_}\"")
        print(f"Train Expected: \"{exp_}\"")
        print("----"*40)

    # Calculate metrics
    epoch_accuracy = (total_correct / total_samples) * 100.0
    print(f"Training Accuracy ({epoch}): {epoch_accuracy}")

    return avg_mse_loss / total_batches, avg_ce_loss / total_batches, epoch_accuracy

@torch.no_grad()
def validate(epoch):
    avg_mse_loss = 0
    avg_ce_loss = 0
    total_samples = 0
    total_correct = 0
    total_batches = 0

    model.eval()
    embedding.eval()
    decoder.eval()

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
        # Initialize sequence with <SOS> token
        batch_size, max_seq_len = target_words.size()
        sos_token = torch.full(
            (batch_size, 1),
            char_to_idx['<sos>'],
            dtype=torch.long,
            device=device
        )

        # Initial decoder input is just <SOS>
        seq_out = sos_token
        decoder_input = embedding(seq_out)

        # Cache encoder output
        src_pos = pos_enc(input_emb)
        src_mask = model.generate_square_subsequent_mask(input_emb.size(1)).to(device)
        memory = model.encoder(src_pos, mask=src_mask, is_causal=True)

        # Generate sequence autoregressively
        for t in range(max_seq_len - 1):  # -1 because we start with <SOS>
            # Add positional encodings
            tgt_pos = pos_enc(decoder_input)

            # Create masks
            tgt_mask = model.generate_square_subsequent_mask(decoder_input.size(1)).to(device)

            # Forward pass
            output_emb = model.decoder(
                tgt_pos,
                memory,
                tgt_mask=tgt_mask,
                tgt_is_causal=True,
            )

            # Decode to vocabulary space
            output_logits = decoder(output_emb[:, -1:, :])

            # Get predicted token
            y_hat = output_logits.argmax(dim=-1)

            # Append to sequence
            seq_out = torch.cat([seq_out, y_hat], dim=1)
            decoder_input = embedding(seq_out)  # Update decoder input

        # Calculate losses
        output_emb = embedding(seq_out)
        output_logits = decoder(output_emb)

        mse_loss = mse_criterion(output_emb, target_emb[:, :seq_out.size(1)])
        ce_loss = ce_criterion(
            output_logits.view(-1, num_tokens),
            target_words[:, :seq_out.size(1)].contiguous().view(-1)
        )

        # Update metrics
        avg_mse_loss += mse_loss.item()
        avg_ce_loss += ce_loss.item()
        total_batches += 1

        with torch.no_grad():
            output_text, expected_text = decode_output(output_logits, target_words)
            total_correct += compare_outputs(output_text, expected_text)
            total_samples += len(output_text)

    # Display the decoded outputs only for the last step of each epoch
    rand_idx = [_.item() for _ in torch.randint(0, len(output_text), (min(10, len(output_text)),))]
    for i in rand_idx:
        out_ = output_text[i]
        exp_ = expected_text[i]
        print(f"Val Output:   \"{out_}\"")
        print(f"Val Expected: \"{exp_}\"")
        print("----"*40)

        # Calculate metrics
        epoch_accuracy = (total_correct / total_samples) * 100.0
        print(f"Validation Accuracy ({epoch}): {epoch_accuracy}")

        return avg_mse_loss / total_batches, avg_ce_loss / total_batches, epoch_accuracy


if not skip_training:
    for epoch in trange(num_epochs):
        # Train
        train_mse_loss, train_ce_loss, train_acc = train_one_epoch(epoch)
        train_mse_loss_list.append(train_mse_loss)
        train_ce_loss_list.append(train_ce_loss)
        train_acc_list.append(train_acc)

        # Validate
        val_mse_loss, val_ce_loss, val_acc = 0.0, 0.0, 0.0
        if not skip_validation:
            val_mse_loss, val_ce_loss, val_acc = validate(epoch)
        val_mse_loss_list.append(val_mse_loss)
        val_ce_loss_list.append(val_ce_loss)
        val_acc_list.append(val_acc)

    # Save model parameters
    torch.save(model.state_dict(), "results/q2_model.pt")
    torch.save(decoder.state_dict(), "results/q2_decoder.pt")
    torch.save(embedding.state_dict(), "results/q2_embedding.pt")

    # Report & Plot
    train_mse_loss_list = np.array(train_mse_loss_list)
    train_ce_loss_list = np.array(train_ce_loss_list)
    train_acc_list = np.array(train_acc_list)
    val_mse_loss_list = np.array(val_mse_loss_list)
    val_ce_loss_list = np.array(val_ce_loss_list)
    val_acc_list = np.array(val_acc_list)

    print("Final accuracy")
    print(f"Train: {train_acc_list[-1]:1.2f}")
    print(f"Val: {val_acc_list[-1]:1.2f}")
    print("Final losses")
    print(f"Train MSE: {train_mse_loss_list[-1]:1.3f}")
    print(f"Train CE: {train_ce_loss_list[-1]:1.3f}")
    print(f"Val MSE: {val_mse_loss_list[-1]:1.3f}")

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
    fig.savefig("results/plots/q2_results.png", dpi=300)
    plt.close()


# Make test predictions
@torch.no_grad()
def predict_test_set():
    predictions = []
    return predictions

# Load the best checkpoints
model.load_state_dict(torch.load("results/q2_model.pt"))
decoder.load_state_dict(torch.load("results/q2_decoder.pt"))
embedding.load_state_dict(torch.load("results/q2_embedding.pt"))

# Set models to evaluation mode
embedding.eval()
model.eval()
decoder.eval()

# Validate
if skip_validation:  # (Skipped during training loop)
    val_mse_loss, val_ce_loss, val_acc = validate(epoch=0)
    print(val_mse_loss, val_ce_loss, val_acc * 100)

"""
# Generate predictions
test_predictions = predict_test_set()

# Save predictions to q1_test.txt
with open('results/q2_test.txt', 'w') as f:
    for pred in test_predictions:
        f.write(f"{pred}\n")
"""