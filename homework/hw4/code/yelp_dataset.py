import json
import torch
from torch.utils.data import Dataset

class YelpDataset(Dataset):
    def __init__(self, split):
        self.split = split
        emb_dim = 50
        # TODO: Load the modified GloVe embeddings
        glove_embs = None

        # TODO: Create a dictionary mapping words to their index in the
        # GloVe embeddings. Remember to add the words in the order as
        # they appear in the dictionary since the same order will be
        # followed in nn.Embedding.from_pretrained used in the main file.
        self.word_indices = {}
        
        # TODO: Load the Yelp dataset and fill in self.reviews and self.stars
        self.reviews = []
        self.stars = []

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        # TODO: Return the word indices for each word in the review
        # text. No need for <EOS> or <SOS> tokens.
        emb = torch.tensor([self.word_indices[_] for _ in self.reviews[idx].split(" ")])
        if self.split == "test":
            return emb
        else:
            return emb, self.stars[idx]

