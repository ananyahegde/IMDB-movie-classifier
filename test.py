import os 
from collections import Counter
from definitions import ROOT_DIR
from src.inputs.load import Load
from src.features.label_encoder import labelEncoder

os.chdir(ROOT_DIR)

path_to_train_pos_data = 'data/raw/train/pos'
path_to_train_neg_data = 'data/raw/train/neg'

load = Load()
raw_train_pos, train_pos_labels = load.load_data(path_to_train_pos_data)
raw_train_neg, train_neg_labels = load.load_data(path_to_train_neg_data)

print(raw_train_pos[0])
print(train_pos_labels[0])

labels = train_pos_labels + train_neg_labels

encoder = labelEncoder()
labels = [encoder.map_label(label) for label in labels]

print(labels[0])

c = Counter(labels)
print(c)

path_to_test_pos_data = 'data/raw/test/pos'
path_to_test_neg_data = 'data/raw/test/neg'

raw_test_pos, test_pos_labels = load.load_data(path_to_test_pos_data)
raw_test_neg, test_neg_labels = load.load_data(path_to_test_neg_data)

test_labels = test_pos_labels + test_neg_labels

encoder = labelEncoder()
test_labels = [encoder.map_label(label) for label in test_labels]

ct = Counter(test_labels)
print(ct)


"""class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out"""