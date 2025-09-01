class labelEncoder:
    r"""Converts multi-class labels into binary labels (0 and 1),
    as required by some PyTorch neural network modules."""

    def map_label(self, score):
        if score <= 4:
            return 0  # negative
        elif score >= 7:
            return 1 # positive
        else:
            return 2