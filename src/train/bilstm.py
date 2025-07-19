import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

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
from src.features.mapping import labelEncoder
from src.features.tokens import Tokenize

os.chdir(ROOT_DIR)
torch.manual_seed(42)
np.random.seed(42)

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


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(0.4)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embed_dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


if not (os.path.exists('models/BiLSTM_SK5Fold_best.pth') and os.path.getsize('models/BiLSTM_SK5Fold_best.pth') > 0):
    print("No pre-trained model available.")
    print("\n-----------Training the model-----------")

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
        sample, label = neg_train_data[i]
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

    splits = 5
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    EPOCHS = 30

    train_accuracies = []
    train_losses = []

    val_accuracies = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_state = None

    for fold, (train_idx, val_idx) in (enumerate(skf.split(padded, labels))):
        print(f"\nFold: {fold+1}")

        train_fold_accuracies = []
        train_fold_losses = []

        val_fold_accuracies = []
        val_fold_losses = []

        x_train, x_val = padded[train_idx], padded[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = BiLSTM(
            vocab_size=len(vocab),
            embed_dim=100,
            hidden_dim=64,
            output_dim=2,
            pad_idx=vocab_to_idx['<pad>']
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8)

        early_stopping = EarlyStopping(patience=8, min_delta=0.01)

        for epoch in range(EPOCHS):
            model.train()

            num_correct = 0
            num_samples = 0
            epoch_loss = 0

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
                num_correct += (predictions == batch_y).sum().item()
                num_samples += predictions.size(0)
                epoch_loss += loss.item()

                acc = round(float(num_correct)/float(num_samples)*100, 2)
                loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
                loop.set_postfix(accuracy=acc, loss=loss.item())

            avg_train_loss = epoch_loss / len(train_loader)
            train_fold_accuracies.append(acc)
            train_fold_losses.append(avg_train_loss)

            # validation
            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predictions = outputs.max(1)
                    val_correct += (predictions == batch_y).sum().item()
                    val_total += batch_y.size(0)

            val_acc = round(float(val_correct) / val_total * 100, 2)
            avg_val_loss = val_loss / len(val_loader)

            # Store validation metrics for this epoch
            val_fold_accuracies.append(val_acc)
            val_fold_losses.append(avg_val_loss)

            # Use average loss for scheduler and early stopping
            scheduler.step(avg_val_loss)
            curr_lr = scheduler.get_last_lr()

            print(f"Validation Accuracy: {val_acc} | Validation Loss: {avg_val_loss:.4f} | Learning Rate: {curr_lr}\n")

            # Use average loss for early stopping and best model tracking
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()

            early_stopping(avg_val_loss)

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        train_accuracies.append(train_fold_accuracies)
        train_losses.append(train_fold_losses)

        val_accuracies.append(val_fold_accuracies)
        val_losses.append(val_fold_losses)

    with open('data/interim/metrics.pkl', 'wb') as file:
        pickle.dump({
            "train_accuracies": train_accuracies,
            "train_losses": train_losses,
            "val_accuracies": val_accuracies,
            "val_losses": val_losses
        }, file)

    torch.save(best_model_state, 'models/BiLSTM_SK5Fold_best.pth')
    print("Best model saved to models/BiLSTM_SK5Fold_best.pth")


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
    sample, label = neg_test_data[i]
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
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = BiLSTM(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=64,
    output_dim=2,
    pad_idx=vocab_to_idx['<pad>']
)

model.load_state_dict(torch.load('models/BiLSTM_SK5Fold_best.pth'))
model = model.to(device)
model.eval()

def test_model(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    test_num_correct = 0
    test_num_samples = 0

    test_accuracies = []
    test_losses = []

    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        for batch_idx, (text, labels) in loop:
            text = text.to(device)
            labels = labels.to(device)

            outputs = model(text)

            loss = criterion(outputs, labels)

            _, predictions = outputs.max(1)
            test_num_correct += (predictions == labels).sum()
            test_num_samples += predictions.size(0)
            acc = round(float(test_num_correct)/float(test_num_samples)*100, 2)

            test_accuracies.append(acc)
            test_losses.append(loss.item())

            loop.set_postfix(accuracy = float(acc),
                            loss = loss.item())

    print(f'Correct predictions: {test_num_correct}/{test_num_samples}')

    with open('data/interim/test_metrics.pkl', 'wb') as file:
        pickle.dump({
            "test_accuracies": test_accuracies,
            "test_losses": test_losses
        }, file)


model = model.to(device)

print("testing...")
test_model(model, test_loader)
 