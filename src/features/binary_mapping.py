class labelEncoder:
    def map_label(self, score):
        if score <= 4:
            return 0  # negative
        elif score >= 7:
            return 1 # positive
        else:
<<<<<<< HEAD:src/features/mapping.py
            return 2
=======
            return 2 # Neutral
        
>>>>>>> parent of 8bd9ae5 (final training regime):src/features/binary_mapping.py
