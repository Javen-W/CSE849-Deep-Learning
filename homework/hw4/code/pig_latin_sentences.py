import json
import torch

class PigLatinSentences(torch.utils.data.Dataset):
    def __init__(self, split, char_to_idx):
        self.char_to_idx = char_to_idx
        self.english_sentences = []
        self.pig_latin_sentences = []

        # TODO: Load the data from the file to self.english_sentences
        # and self.pig_latin_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        # TODO: Load corresponding english and pig latin sentences,
        # append <sos> and <eos> tokens, convert them to indices using
        # char_to_idx, and return the indices.
        return eng_word_idx, pig_latin_word_idx
        