from torch import cuda
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
skip_validation = False

# Seed
torch.manual_seed(777)

# Hyperparameters
batch_size = 64
lr = 0.002
n_epochs = 50
n_workers = 0
n_tokens = 30
emb_dim = 100
n_head = 2
n_encoder_layers = 2
n_decoder_layers = 2
dim_feedforward = 128
dropout = 0.1
batch_first = False

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

pad_idx = char_to_idx['<pad>']
eos_idx = char_to_idx['<eos>']

# reverse, integer to character mapping
idx_to_char = {}
for char, idx in char_to_idx.items():
    idx_to_char[idx] = char

@torch.no_grad()
def decode_output(output, target_words, is_seq_out=False):
    if is_seq_out:
        out_words = output.detach().cpu().numpy()  # seq_out: (seq_len, batch_size)
    else:
        out_words = output.argmax(dim=-1).detach().cpu().numpy()  # logits: (seq_len, batch_size, vocab_size)
    target_words = target_words.detach().cpu().numpy()
    out_decoded, exp_decoded = [], []
    pad_pos = char_to_idx['<pad>']
    for i in range(output.size(1)):
        out_str = "".join([idx_to_char[idx] for idx in out_words[:, i] if idx != pad_pos])
        if "<eos>" in out_str:
            out_str = out_str.split("<eos>")[0] + "<eos>"
        out_decoded.append(out_str)
        exp_decoded.append("".join([idx_to_char[idx] for idx in target_words[:, i] if idx != pad_pos]))
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
    # Extract raw sequences
    eng_batch, pig_batch = zip(*batch)

    # Prepare input_sequence
    eng_padded = pad_sequence(eng_batch, batch_first=batch_first, padding_value=char_to_idx['<pad>']).to(device)
    input_sequence = embedding(eng_padded)

    if pig_batch[0] is not None:
        # Prepare output_sequence
        pig_padded = pad_sequence(pig_batch, batch_first=batch_first, padding_value=char_to_idx['<pad>']).to(device)
        output_sequence = embedding(pig_padded)
        return input_sequence, output_sequence, pig_padded
    else:
        return input_sequence

def word_accuracy(output_text, expected_text):
    total_words = 0
    correct_words = 0
    for out, exp in zip(output_text, expected_text):
        out_words = out.replace("<sos>", "").split("<eos>")[0].split()
        exp_words = exp.replace("<sos>", "").split("<eos>")[0].split()
        total_words += len(exp_words)
        correct_words += sum(1 for o, e in zip(out_words, exp_words) if o == e)
    return correct_words / total_words * 100 if total_words > 0 else 0

def write_metrics_to_file(epoch, train_mse, train_ce, train_acc, val_mse, val_ce, val_acc, final=False):
    # Determine mode: 'w' for first write only, 'a' for appending
    mode = 'w' if epoch == 0 and not final else 'a'
    with open('results/q2_history.txt', mode) as _f:
        if final:
            # Write final metrics
            _f.write("\nFinal accuracies\n")
            _f.write(f"Train: {train_acc:1.2f}\n")
            _f.write(f"Val: {val_acc:1.2f}\n")
            _f.write("\nFinal losses\n")
            _f.write(f"Train MSE: {train_mse:1.3f}\n")
            _f.write(f"Train CE: {train_ce:1.3f}\n")
            _f.write(f"Val MSE: {val_mse:1.3f}\n")
            _f.write(f"Val CE: {val_ce:1.3f}\n")
        else:
            # Write per-epoch metrics
            _f.write(
                f"[{epoch + 1}/{n_epochs}] Training MSE: {train_mse:.4f}, CE: {train_ce:.4f}, Accuracy: {train_acc:.2f}%\n")
            _f.write(
                f"[{epoch + 1}/{n_epochs}] Validation MSE: {val_mse:.4f}, CE: {val_ce:.4f}, Accuracy: {val_acc:.2f}%")
            _f.write("\n")

# Create Datasets
train_dataset = PigLatinSentences("train", char_to_idx)
val_dataset = PigLatinSentences("val", char_to_idx)
test_dataset = PigLatinSentences("test", char_to_idx)

# TODO: Define your embedding
embedding = nn.Embedding(
    num_embeddings=n_tokens,
    embedding_dim=emb_dim,
    padding_idx=char_to_idx['<pad>'],
)
embedding = embedding.to(device)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=n_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=n_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=n_workers)

# TODO: Create your Transformer model
model = nn.Transformer(
    d_model=emb_dim,
    nhead=n_head,
    num_encoder_layers=n_encoder_layers,
    num_decoder_layers=n_decoder_layers,
    dim_feedforward=dim_feedforward,
    batch_first=batch_first,
    dropout=dropout,
)
model = model.to(device)

# TODO: Create your decoder from embedding space to the vocabulary space
decoder = nn.Linear(
    in_features=emb_dim,
    out_features=n_tokens,
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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
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

    for input_emb, target_emb, target_words in tqdm(train_loader, leave=False, desc=f"Train epoch {epoch+1}/{n_epochs}"):
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
        src_pos = pos_enc(input_emb)
        tgt_input = target_emb[:-1, :]  # All but last token
        tgt_output = target_words[1:, :]  # All but first token
        tgt_pos = pos_enc(tgt_input)

        # Create masks
        src_mask = model.generate_square_subsequent_mask(src_pos.size(0)).to(device)
        tgt_mask = model.generate_square_subsequent_mask(tgt_pos.size(0)).to(device)

        # Scheduled sampling
        """
        prob = max(0.0, min(0.9, (epoch - 5) * 0.02))  # Up to 90% by epoch 50
        if torch.rand(1).item() < prob:
            temp_tgt_pos = pos_enc(tgt_input)
            temp_output_emb = model(src=src_pos, tgt=temp_tgt_pos, src_mask=None, tgt_mask=tgt_mask)
            temp_logits = decoder(temp_output_emb)
            next_token = temp_logits.argmax(dim=-1)
            tgt_input = embedding(next_token)
        tgt_pos = pos_enc(tgt_input)
        """

        # Forward pass
        output_emb = model(
            src=src_pos,
            tgt=tgt_pos,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_is_causal=False,
            tgt_is_causal=True,
        )

        # Decode to vocabulary space
        output_logits = decoder(output_emb)

        # Calculate the losses
        mse_loss = mse_criterion(output_emb, tgt_input)
        ce_loss = ce_criterion(output_logits.view(-1, n_tokens), tgt_output.view(-1))

        # Update the model parameters
        total_loss = 1.0 * mse_loss + ce_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # Update metrics
        avg_mse_loss += mse_loss.item()
        avg_ce_loss += ce_loss.item()
        total_batches += 1

        with torch.no_grad():
            output_text, expected_text = decode_output(
                output=output_logits,
                target_words=target_words,
                is_seq_out=False,
            )
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
    avg_mse = avg_mse_loss / total_batches
    avg_ce = avg_ce_loss / total_batches
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    print(f"Training MSE: {avg_mse:.4f}, CE: {avg_ce:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_mse, avg_ce, accuracy

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

    for input_emb, target_emb, target_words in tqdm(val_loader, leave=False, desc=f"Val epoch {epoch+1}/{n_epochs}"):
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
        max_seq_len, batch_size = target_words.size()
        sos_token = torch.full((1, batch_size), char_to_idx['<sos>'], device=device)
        seq_out = sos_token
        logits_list = []  # Store logits for each step

        # Cache encoder output
        src_pos = pos_enc(input_emb)
        src_mask = model.generate_square_subsequent_mask(input_emb.size(0)).to(device)
        memory = model.encoder(src_pos, mask=src_mask, is_causal=False)

        # Generate sequence autoregressively
        for t in range(max_seq_len - 1):  # -1 because we start with <SOS>
            # Prepare decoder input
            decoder_input = embedding(seq_out)
            tgt_pos = pos_enc(decoder_input)
            tgt_mask = model.generate_square_subsequent_mask(tgt_pos.size(0)).to(device)

            # Forward pass
            output_emb = model.decoder(
                tgt=tgt_pos,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_is_causal=True,
            )

            # Decode to vocabulary space
            output_logits = decoder(output_emb[-1:, :, :]) # (1, batch_size, n_tokens)
            logits_list.append(output_logits)  # Store logits for CE loss

            # Get predicted token
            y_hat = output_logits.argmax(dim=-1)

            # Append to sequence
            seq_out = torch.cat([seq_out, y_hat], dim=0)

        # Compute losses
        output_emb = embedding(seq_out)
        mse_loss = mse_criterion(output_emb, target_emb[:seq_out.size(0), :, :])
        # Concatenate logits and targets for CE loss
        logits = torch.cat(logits_list, dim=0)  # (seq_len-1, batch_size, n_tokens)
        ce_loss = ce_criterion(
            logits.view(-1, n_tokens),
            target_words[1:seq_out.size(0), :].contiguous().view(-1)  # Align with logits length
        )

        # Update metrics
        avg_mse_loss += mse_loss.item()
        avg_ce_loss += ce_loss.item()
        total_batches += 1

        with torch.no_grad():
            output_text, expected_text = decode_output(
                output=logits,
                target_words=target_words,
                is_seq_out=False,
            )
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
    avg_mse = avg_mse_loss / total_batches
    avg_ce = avg_ce_loss / total_batches
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    word_acc = word_accuracy(output_text, expected_text)
    print(f"Validation MSE: {avg_mse:.4f}, CE: {avg_ce:.4f}, Accuracy: {accuracy:.2f}, Word-level Accuracy: {word_acc:.2f}%")

    # Progress scheduler
    scheduler.step(avg_ce)

    return avg_mse, avg_ce, accuracy, word_acc

if not skip_training:
    for epoch in trange(n_epochs):
        # Train
        train_mse_loss, train_ce_loss, train_acc = train_one_epoch(epoch)
        train_mse_loss_list.append(train_mse_loss)
        train_ce_loss_list.append(train_ce_loss)
        train_acc_list.append(train_acc)

        # Validate
        if not skip_validation:
            val_mse_loss, val_ce_loss, val_acc, _ = validate(epoch)
        else:
            val_mse_loss, val_ce_loss, val_acc = 0.0, 0.0, 0.0
        val_mse_loss_list.append(val_mse_loss)
        val_ce_loss_list.append(val_ce_loss)
        val_acc_list.append(val_acc)

        # Write metrics to file
        write_metrics_to_file(
            epoch=epoch,
            train_mse=train_mse_loss,
            train_ce=train_ce_loss,
            train_acc=train_acc,
            val_mse=val_mse_loss,
            val_ce=val_ce_loss,
            val_acc=val_acc,
            final=False,
        )

    # Save model parameters
    save_dict = {
        "transformer": model.state_dict(),
        "decoder": decoder.state_dict(),
        "embedding": embedding.state_dict()
    }
    torch.save(save_dict, "results/q2_model.pt")

    # Report & Plot
    train_mse_loss_list = np.array(train_mse_loss_list)
    train_ce_loss_list = np.array(train_ce_loss_list)
    train_acc_list = np.array(train_acc_list)
    val_mse_loss_list = np.array(val_mse_loss_list)
    val_ce_loss_list = np.array(val_ce_loss_list)
    val_acc_list = np.array(val_acc_list)

    # Print and write final metrics
    print("Final accuracy")
    print(f"Train: {train_acc_list[-1]:1.2f}")
    print(f"Val: {val_acc_list[-1]:1.2f}")
    print("Final losses")
    print(f"Train MSE: {train_mse_loss_list[-1]:1.3f}")
    print(f"Train CE: {train_ce_loss_list[-1]:1.3f}")
    print(f"Val MSE: {val_mse_loss_list[-1]:1.3f}")

    write_metrics_to_file(
        epoch=0,  # Not used for final write
        train_mse=train_mse_loss_list[-1],
        train_ce=train_ce_loss_list[-1],
        train_acc=train_acc_list[-1],
        val_mse=val_mse_loss_list[-1],
        val_ce=val_ce_loss_list[-1],
        val_acc=val_acc_list[-1],
        final=True
    )

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].plot(np.arange(n_epochs), train_ce_loss_list + train_mse_loss_list, label="Train")
    axs[0, 0].plot(np.arange(n_epochs), val_ce_loss_list + val_mse_loss_list, label="Val")
    axs[0, 0].legend()
    axs[0, 0].set_title("Total Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_yscale("log")

    axs[0, 1].plot(np.arange(n_epochs), train_acc_list, label="Train")
    axs[0, 1].plot(np.arange(n_epochs), val_acc_list, label="Val")
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Accuracy (%)")

    axs[1, 0].plot(np.arange(n_epochs), train_mse_loss_list, label="Train")
    axs[1, 0].plot(np.arange(n_epochs), val_mse_loss_list, label="Val")
    axs[1, 0].legend()
    axs[1, 0].set_title("MSE Loss")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].set_yscale("log")

    axs[1, 1].plot(np.arange(n_epochs), train_ce_loss_list, label="Train")
    axs[1, 1].plot(np.arange(n_epochs), val_ce_loss_list, label="Val")
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
    model.eval()
    embedding.eval()
    decoder.eval()

    for input_emb, _, _ in tqdm(test_loader, leave=False, desc="Generating test predictions"):
        max_seq_len, batch_size = input_emb.size(0), input_emb.size(1)  # (seq_len, batch_size, emb_dim)
        sos_token = torch.full((1, batch_size), char_to_idx['<sos>'], device=device)  # (1, batch_size)
        seq_out = sos_token

        # Cache encoder output
        src_pos = pos_enc(input_emb)
        memory = model.encoder(src_pos, mask=None, is_causal=False)

        # Generate sequence autoregressively
        for t in range(max_seq_len - 1):  # -1 because we start with <sos>
            decoder_input = embedding(seq_out)
            tgt_pos = pos_enc(decoder_input)
            tgt_mask = model.generate_square_subsequent_mask(tgt_pos.size(0)).to(device)
            output_emb = model.decoder(
                tgt=tgt_pos,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_is_causal=True,
            )
            output_logits = decoder(output_emb[-1:, :, :])  # (1, batch_size, n_tokens)
            y_hat = output_logits.argmax(dim=-1)  # (1, batch_size)
            seq_out = torch.cat([seq_out, y_hat], dim=0)  # (seq_len_so_far+1, batch_size)

        # Decode the generated sequences
        output_text, _ = decode_output(seq_out, seq_out, is_seq_out=True)  # We don't need target_words
        predictions.extend(output_text)

    return predictions

# Load the best checkpoint
checkpoint = torch.load("results/q2_model.pt", weights_only=True)
model.load_state_dict(checkpoint["transformer"])
decoder.load_state_dict(checkpoint["decoder"])
embedding.load_state_dict(checkpoint["embedding"])

# Set models to evaluation mode
embedding.eval()
model.eval()
decoder.eval()

# Validate
if skip_validation:  # (Skipped during training loop)
    val_mse_loss, val_ce_loss, val_acc = validate(epoch=0)

# Generate predictions
test_predictions = predict_test_set()

# Save predictions
with open('results/q2_test.txt', 'w') as f:
    for pred in test_predictions:
        f.write(f"{pred}\n")