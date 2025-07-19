import torch
from torch.nn.utils.rnn import pad_sequence

class Tokenize():
    def __init__(self):
        print("\ntokenizing...")

    def tokenize_data(self, samples, vocab=None, vocab_to_idx=None, train=True, max_length=300):
        if not vocab and not vocab_to_idx:
            word_freq = {}
            for review in samples:
                for word in review:
                    word_freq[word] = word_freq.get(word, 0) + 1

            MAX_VOCAB_SIZE = 20000
            sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            sorted_vocab = sorted_vocab[:MAX_VOCAB_SIZE]
            vocab = ['<pad>', '<unk>'] + [word for word, _ in sorted_vocab]
            vocab_to_idx = {word: idx for idx, word in enumerate(vocab)}

        numericalized = [[vocab_to_idx.get(token, vocab_to_idx['<unk>']) for token in review] for review in samples]

        truncated = []
        for seq in numericalized:
            if len(seq) > max_length:
                truncated.append(seq[:max_length])
            else:
                truncated.append(seq)

        tensorized = [torch.tensor(seq) for seq in truncated]
        padded = pad_sequence(tensorized, batch_first=True, padding_value=vocab_to_idx['<pad>'])

        print("Tokenization done.")
        if train:
            return padded, vocab, vocab_to_idx

        else:
            return padded