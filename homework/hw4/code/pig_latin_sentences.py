import json
import torch

class PigLatinSentences(torch.utils.data.Dataset):
    def __init__(self, split, char_to_idx):
        self.char_to_idx = char_to_idx
        self.english_sentences = []
        self.pig_latin_sentences = []
        self.split = split
        self.sos = [self.char_to_idx['<sos>']]
        self.eos = [self.char_to_idx['<eos>']]

        # TODO: Load the data from the file to self.english_sentences and self.pig_latin_sentences
        # Load the appropriate text file based on the split
        data_file = f'code/data/reviews_pig_latin_data_{self.split}.txt'
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue
                # Parse each line as a separate JSON object
                item = json.loads(line.strip())
                self.english_sentences.append(item['original'])
                if split != 'test':
                    self.pig_latin_sentences.append(item['pig_latin'])

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        # TODO: Load corresponding english and pig latin sentences,
        # append <sos> and <eos> tokens, convert them to indices using char_to_idx, and return the indices.
        eng_word_idx = torch.tensor(self.sos + [self.char_to_idx[c] for c in self.english_sentences[idx]] + self.eos)
        if self.split == "test":
            return eng_word_idx, None
        else:
            pig_latin_word_idx = torch.tensor(
                self.sos + [self.char_to_idx[c] for c in self.pig_latin_sentences[idx]] + self.eos)
            return eng_word_idx, pig_latin_word_idx