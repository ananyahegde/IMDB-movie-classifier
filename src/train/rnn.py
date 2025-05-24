import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader
from definitions import ROOT_DIR
from src.inputs.load import Load
from src.inputs.preprocess import Preprocess
from src.features.count_vectorizer import countVectorizer
from src.features.label_encoder import labelEncoder

os.chdir(ROOT_DIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device = {device}")

path_to_train_pos_data = 'data/raw/train/pos'
path_to_train_neg_data = 'data/raw/train/neg'

load = Load()
raw_train_pos, train_pos_labels = load.load_data(path_to_train_pos_data)
raw_train_neg, train_neg_labels = load.load_data(path_to_train_neg_data)

labels = train_pos_labels + train_neg_labels

encoder = labelEncoder()
train_labels = np.array([encoder.map_label(label) for label in labels])

preprocess = Preprocess()
processed_train_pos = preprocess.process_data(raw_train_pos)
processed_train_neg = preprocess.process_data(raw_train_neg)

processed_train_data = processed_train_pos + processed_train_neg
processed_train_data, train_labels = shuffle(processed_train_data, train_labels)

vectorizer = countVectorizer()
best_count_vectorizer, train_features = vectorizer.count_vectorizer(processed_train_data, train_labels)

path_to_test_pos_data = 'data/raw/test/pos'
path_to_test_neg_data = 'data/raw/test/neg'

raw_test_pos, test_pos_labels = load.load_data(path_to_test_pos_data)
raw_test_neg, test_neg_labels = load.load_data(path_to_test_neg_data)

test_labels = test_pos_labels + test_neg_labels
test_labels = np.array([encoder.map_label(label) for label in test_labels])

processed_test_pos = preprocess.process_data(raw_test_pos)
processed_test_neg = preprocess.process_data(raw_test_neg)

processed_test_data = processed_test_pos + processed_test_neg
processed_test_data, test_labels = shuffle(processed_test_data, test_labels)

test_features = best_count_vectorizer.transform(processed_test_data)

# train_x = train_features.toarray().astype(np.float32)
# train_y = train_labels.astype(np.int64)
# test_x = test_features.toarray().astype(np.float32)
# test_y = test_labels.astype(np.int64)
#
# batch_size = 128
# trainset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
# testset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
#
# trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
# testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)
#
# dataiter = iter(trainloader)
# x, y = next(dataiter)
#
# print('Sample batch size: ', x.size())
# print('Sample batch input: \n', x)
# print()
# print('Sample label size: ', y.size())
# print('Sample label input: \n', y)
#

# class IMDBDataset(Dataset):
#     def __init__(self, data):
#         self.texts = data['text'].values
#         self.labels = data['label'].values
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
#         return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)
#
#
# class SentimentRNN(nn.Module):
#     def __init__(self, vocab_size, embed_size, hidden_size, output_size):
#         super(SentimentRNN, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         x = self.embedding(x)
#         h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
#         out, _ = self.rnn(x, h0)
#         out = self.fc(out[:, -1, :])
#         return out
#
# vocab_size = len(vocab) + 1
# embed_size = 128
# hidden_size = 128
# output_size = 2
# model = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)
#
#
#
# train_dataset = SentimentDataset(train_data)
# test_dataset = SentimentDataset(test_data)
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     for texts, labels in train_loader:
#         outputs = model(texts)
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         epoch_loss += loss.item()
#
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
#
#
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for texts, labels in test_loader:
#         outputs = model(texts)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# accuracy = 100 * correct / total
# print(f'Accuracy: {accuracy:.2f}%')
#
