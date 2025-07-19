import os
import pickle
from definitions import ROOT_DIR
import matplotlib.pyplot as plt

os.chdir(ROOT_DIR)

with open('data/interim/metrics.pkl', 'rb') as file:
    acc_loss = pickle.load(file)

train_acc = acc_loss['train_accuracies']
val_acc = acc_loss['val_accuracies']

epochs = [[i for i in range(len(run))] for run in train_acc]

n_runs = len(train_acc)
cols = 2
rows = (n_runs + 1) // cols

fig, axs = plt.subplots(rows, cols, figsize=(10, 4 * rows))
axs = axs.ravel()

for i, (ep, train, val) in enumerate(zip(epochs, train_acc, val_acc)):
    axs[i].plot(ep, train, label='Train')
    axs[i].plot(ep, val, label='Val')
    axs[i].set_xlim(1, 30)
    axs[i].set_title(f'Run {i+1}')
    axs[i].set_xlabel('Epoch')
    axs[i].set_ylabel('Accuracy')
    axs[i].grid(True)
    axs[i].legend()

for j in range(i + 1, len(axs)):
    axs[j].set_visible(False)

plt.tight_layout()
plt.show()
