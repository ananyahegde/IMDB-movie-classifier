import os
import re
from concurrent.futures import ThreadPoolExecutor

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_data(dir_path) -> list[str]:
    filenames = os.listdir(dir_path)
    labels = [int(re.search(r'_(\d+)\.', f).group(1)) for f in filenames]

    filepaths = [os.path.join(dir_path, f) for f in filenames]
    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(read_file, filepaths))
    return texts, labels