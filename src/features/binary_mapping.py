class labelEncoder:
    def map_label(self, score):
        if score <= 4:
            return 0  # negative
        elif score >= 7:
            return 1 # positive
        else:
            return 2 # Neutral
        