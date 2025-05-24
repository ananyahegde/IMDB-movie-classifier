class labelEncoder:
    def map_label(self, score):
        if score <= 2:
            return 0  # very bad
        elif score <= 4:
            return 1  # bad
        elif score <= 6:
            return 2  # neutral
        elif score <= 8:
            return 3  # good
        else:
            return 4  # very good
