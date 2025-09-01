import torch
from torch.nn.utils.rnn import pad_sequence

class Tokenize():
    r"""Converts preprocessed text tokens into numerical tensors for neural network training.

    This class handles the conversion of tokenized text data into numerical format suitable for
    deep learning models. It creates vocabulary mappings, converts tokens to indices, handles
    sequence truncation and padding, and returns PyTorch tensors ready for model input.

    The tokenization pipeline includes:
        - Vocabulary creation from training data (for training mode)
        - Token-to-index mapping using existing vocabulary (for inference mode)
        - Sequence truncation to maximum length
        - Tensor conversion and padding for batch processing

    **Methods**:

    - `__init__()`: Initializes the tokenizer and prints a status message.
    - `tokenize_data(samples, vocab=None, vocab_to_idx=None, train=True, max_length=300)`:
      Converts tokenized text samples into padded numerical tensors.

    **Returns**:
    - If `train=True`: `(padded_tensors, vocab, vocab_to_idx)`
        - `padded_tensors (torch.Tensor)`: Batch of padded sequences with shape (batch_size, max_seq_length)
        - `vocab (list[str])`: Complete vocabulary list including special tokens
        - `vocab_to_idx (dict)`: Token-to-index mapping dictionary
    - If `train=False`: `padded_tensors (torch.Tensor)`
        - `padded_tensors (torch.Tensor)`: Batch of padded sequences with shape (batch_size, max_seq_length)

    **Special Tokens**:
    - `<pad>` (index 0): Used for padding shorter sequences
    - `<unk>` (index 1): Used for out-of-vocabulary tokens

    **Vocabulary Creation**:
    - Maximum vocabulary size: 20,000 tokens (excluding special tokens)
    - Vocabulary is built from token frequency in training samples
    - Most frequent tokens are prioritized for inclusion
    """

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
            vocab_to_idx = {word:idx for idx, word in enumerate(vocab)}

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