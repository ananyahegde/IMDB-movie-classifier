import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

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
from torchvision import models

from definitions import ROOT_DIR
from src.inputs.load import Load
from src.inputs.preprocess import Preprocess
from src.features.count_vectorizer import countVectorizer
from src.features.label_encoder import labelEncoder
from src.features.tokens import Tokenize

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


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])


if not (os.path.exists('models/BiLSTM.pth') and os.path.getsize('models/BiLSTM.pth') > 0):
    print("No pre-trained model available.")
    print("-----------Training the model-----------")

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

    tokenize = Tokenize()
    padded, vocab, vocab_to_idx = tokenize.tokenize_data(train_samples)

    with open('data/interim/vocab.pkl', 'wb') as f:
        pickle.dump((vocab, vocab_to_idx), f)

    encoder = labelEncoder()
    train_labels = [encoder.map_label(label) for label in train_labels]
    labels = torch.tensor(train_labels, dtype=torch.long)

    dataset = TensorDataset(padded, labels)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = BiLSTMModel(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_dim=128,
        output_dim=5,
        pad_idx=vocab_to_idx['<pad>']
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    EPOCHS = 5

    num_correct = 0
    num_samples = 0

    for epoch in range(EPOCHS):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for batch_idx, (batch_x, batch_y) in loop:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

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

    torch.save(model.state_dict(), 'models/BiLSTM.pth')

# Testing

print("-----------Testing the pre-trained model-----------")
path_to_test_pos_data = 'data/raw/test/pos'
path_to_test_neg_data = 'data/raw/test/neg'

pos_test_data = IMDB(path_to_test_pos_data, transform=None)
neg_test_data = IMDB(path_to_test_neg_data, transform=None)


test_samples = []
test_labels = []

for i in range(len(pos_test_data)):
    sample, label = pos_test_data[i]
    test_samples.append(sample)
    test_labels.append(label)

for i in range(len(neg_test_data)):
    sample, label = pos_test_data[i]
    test_samples.append(sample)
    test_labels.append(label)


preprocess = Preprocess()
test_samples = preprocess.process_data(test_samples)

with open('data/interim/vocab.pkl', 'rb') as f:
    vocab, vocab_to_idx = pickle.load(f)

tokenize = Tokenize()
padded = tokenize.tokenize_data(test_samples, vocab, vocab_to_idx, train=False)

encoder = labelEncoder()
test_labels = [encoder.map_label(label) for label in test_labels]
labels = torch.tensor(test_labels, dtype=torch.long)

dataset = TensorDataset(padded, labels)
test_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = BiLSTMModel(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=128,
    output_dim=5,
    pad_idx=vocab_to_idx['<pad>']
)

model.load_state_dict(torch.load('models/BiLSTM.pth'))
model.eval()

def test_model(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        for batch_idx, (text, labels) in loop:
            text = text.to(device)
            labels = labels.to(device)

            outputs = model(text)

            loss = criterion(outputs, labels)

            _, predictions = outputs.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
            acc = round(float(num_correct)/float(num_samples)*100, 2)

            loop.set_postfix(accuracy = float(acc),
                            loss = loss.item())

    print(f'Correct predictions: {num_correct}/{num_samples}')

model = model.to(device)

print("testing...")
test_model(model, test_loader)