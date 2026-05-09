import re
import json
import torch
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader


def load_ag_news(data_dir='data'):
    train_df = pd.read_csv(f'{data_dir}/train.csv', header=None, names=['label', 'title', 'description'])
    test_df = pd.read_csv(f'{data_dir}/test.csv', header=None, names=['label', 'title', 'description'])

    # combine title and description
    train_df['text'] = train_df['title'] + ' ' + train_df['description']
    test_df['text'] = test_df['title'] + ' ' + test_df['description']

    # labels are 1-4, shift to 0-3
    train_df['label'] = train_df['label'] - 1
    test_df['label'] = test_df['label'] - 1

    return train_df[['label', 'text']], test_df[['label', 'text']]


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def build_vocab(texts, max_size=25000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    most_common = counter.most_common(max_size - 2)

    vocab = {'<pad>': 0, '<unk>': 1}
    for i, (word, _) in enumerate(most_common):
        vocab[word] = i + 2

    return vocab


def encode_text(text, vocab, max_len):
    tokens = tokenize(text)
    indices = [vocab.get(t, vocab['<unk>']) for t in tokens]

    if len(indices) >= max_len:
        indices = indices[:max_len]
    else:
        indices = indices + [vocab['<pad>']] * (max_len - len(indices))

    return indices


def save_vocab(vocab, path):
    with open(path, 'w') as f:
        json.dump(vocab, f)


def load_vocab(path):
    with open(path, 'r') as f:
        return json.load(f)


class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = encode_text(self.texts[idx], self.vocab, self.max_len)
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def get_dataloaders(train_df, test_df, vocab, max_len, batch_size):
    train_dataset = NewsDataset(
        train_df['text'].tolist(), train_df['label'].tolist(),
        vocab, max_len
    )
    test_dataset = NewsDataset(
        test_df['text'].tolist(), test_df['label'].tolist(),
        vocab, max_len
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
