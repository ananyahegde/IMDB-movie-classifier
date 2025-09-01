import os
import re
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class Load:
    r"""Loads files from a specified directory and extracts review scores embedded in the filenames.

    **methods**:

    - `read_file(filepath) -> str`:
    Reads the given file and returns its content as a cleaned string.

    - `load_data(dir_path) -> list[str]`:
    Retrieves all text files from the specified directory and extracts *labels* (review scores).
    Utilizes multithreading for efficient file reading.
    Returns both the extracted text data and corresponding labels.
    """

    def __init__(self):
        print('\nimporting data...')

    def read_file(self, filepath) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def load_data(self, dir_path) -> list[str]:
        self.t1 = time.perf_counter()

        filenames = os.listdir(dir_path)
        labels = [int(re.search(r'_(\d+)\.', f).group(1)) for f in filenames]

        filepaths = [os.path.join(dir_path, f) for f in filenames]

        with ThreadPoolExecutor() as executor:
            texts = list(tqdm(executor.map(self.read_file, filepaths), total=len(filepaths)))

        self.t2 = time.perf_counter()
        print(f"Finished in {round(self.t2 - self.t1, 4)} second(s).")
        return texts, labels