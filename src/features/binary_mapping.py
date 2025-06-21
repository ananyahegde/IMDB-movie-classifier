class labelEncoder:
    def map_label(self, score):
        if score <= 5:
            return 0  # negative
        else:
            return 1 # positive