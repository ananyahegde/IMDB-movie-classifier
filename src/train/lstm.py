import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from itertools import chain

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Embedding
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision import transforms

from definitions import ROOT_DIR
from src.inputs.load import Load
from src.inputs.preprocess import Preprocess
from src.features.count_vectorizer import countVectorizer
from src.features.label_encoder import labelEncoder
from src.features.word2vec_vectorizer import word2vecVectorizer


os.chdir(ROOT_DIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device = {device}")

class IMDB(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        load = Load()
        self.data, self.labels = load.load_data(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


path_to_train_pos_data = 'data/raw/train/pos'
path_to_train_neg_data = 'data/raw/train/neg'

pos_train_data = IMDB(path_to_train_pos_data, transform=None)
neg_train_data = IMDB(path_to_train_neg_data, transform=None)


train_samples = []
train_labels = []

for i in range(len(pos_train_data)):
    sample, label = pos_train_data[i]
    train_samples.append(sample)
    train_labels.append(label)

for i in range(len(neg_train_data)):
    sample, label = pos_train_data[i]
    train_samples.append(sample)
    train_labels.append(label)


preprocess = Preprocess()
train_samples = preprocess.process_data(train_samples)

encoder = labelEncoder()
train_labels = [encoder.map_label(label) for label in train_labels]


word_freq = {}
for review in train_samples:
    for word in review:
        word_freq[word] = word_freq.get(word, 0) + 1

sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
vocab = ['<pad>', '<unk>'] + [word for word, _ in sorted_vocab]
vocab_to_idx = {word: idx for idx, word in enumerate(vocab)}

numericalized = [[vocab_to_idx.get(token, vocab_to_idx['<unk>']) for token in review] for review in train_samples]
tensorized = [torch.tensor(seq) for seq in numericalized]

padded = pad_sequence(tensorized, batch_first=True, padding_value=vocab_to_idx['<pad>'])

labels = torch.tensor(train_labels, dtype=torch.long)
dataset = TensorDataset(padded, labels)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])


model = LSTMModel(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=128,
    output_dim=5,
    pad_idx=vocab_to_idx['<pad>']
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 2

num_correct = 0
num_samples = 0

for epoch in range(EPOCHS):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for batch_idx, (batch_x, batch_y) in loop:
        scores = model(batch_x)
        loss = criterion(scores, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predictions = scores.max(1)
        num_correct += (predictions == batch_y).sum()
        num_samples += predictions.size(0)
        acc = round(float(num_correct)/float(num_samples)*100, 2)

        loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
        loop.set_postfix(accuracy = float(acc),
                            loss = loss.item())

torch.save(model.state_dict(), "models/LSTM.pth")
