import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

class Load:
    def __init__(self):
        print('importing data...')

    def read_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def load_data(self, dir_path) -> list[str]:
        self.t1 = time.perf_counter()

        filenames = os.listdir(dir_path)
        labels = [int(re.search(r'_(\d+)\.', f).group(1)) for f in filenames]

        filepaths = [os.path.join(dir_path, f) for f in filenames]
        with ThreadPoolExecutor() as executor:
            texts = list(executor.map(self.read_file, filepaths))

        self.t2 = time.perf_counter()
        print(f"Finished in {round(self.t2 - self.t1, 4)} second(s).")
        return texts, labels