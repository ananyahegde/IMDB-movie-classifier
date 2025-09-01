import os
import gc
import sys
import re
import time
import string
import nltk
import psutil
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from definitions import ROOT_DIR

class LoadSimple:
    r"""This class compares how much time it would take to load and preprocess data without multithreading.
    This is for comparison purpose."""

    def __init__(self):
        print('\nimporting data...')

    def read_file(self, filepath) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def load_data(self, dir_path) -> list[str]:
        t1 = time.perf_counter()
        filenames = os.listdir(dir_path)
        labels = [int(re.search(r'_(\d+)\.', f).group(1)) for f in filenames]
        texts = [self.read_file(os.path.join(dir_path, f)) for f in tqdm(filenames, desc=f"Reading {dir_path}")]
        t2 = time.perf_counter()
        print(f"Loaded {len(texts)} files from {dir_path} in {round(t2 - t1, 4)}s")
        return texts, labels

class PreprocessSimple:
    def __init__(self):
        print("\npreprocessing data...")

    def download_resources(self):
        print("Downloading NLTK resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("Finished downloading resources.")

    def preprocess_pipeline(self, text: str) -> list[str]:
        text = text.lower().replace("<br />", "")
        for p in string.punctuation:
            text = text.replace(p, '')
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(t) for t in tokens]

    def process_data(self, texts: list[str]) -> list[list[str]]:
        t1 = time.perf_counter()
        processed = [self.preprocess_pipeline(t) for t in tqdm(texts, desc="Preprocessing texts")]
        t2 = time.perf_counter()
        print(f"Processed {len(texts)} texts in {round(t2 - t1, 4)}s")
        return processed

def run_main():
    os.chdir(ROOT_DIR)

    # Initialize monitoring
    process = psutil.Process()
    start_total = time.perf_counter()
    start_cpu_total = time.process_time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    loader = LoadSimple()
    preprocessor = PreprocessSimple()

    t1 = time.perf_counter()
    raw_pos, _ = loader.load_data('data/raw/train/pos')
    raw_neg, _ = loader.load_data('data/raw/train/neg')
    t2 = time.perf_counter()
    print(f"Data loading finished in {round(t2 - t1, 4)}s")

    t3 = time.perf_counter()
    preprocessor.download_resources()
    t4 = time.perf_counter()
    print(f"Resource download finished in {round(t4 - t3, 4)}s")

    t5 = time.perf_counter()
    preprocessor.process_data(raw_pos)
    preprocessor.process_data(raw_neg)
    t6 = time.perf_counter()
    print(f"Data preprocessing finished in {round(t6 - t5, 4)}s")

    # Final measurements
    end_total = time.perf_counter()
    end_cpu_total = time.process_time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB

    total_time = end_total - start_total
    cpu_time = end_cpu_total - start_cpu_total
    memory_used = end_memory - start_memory
    efficiency = (cpu_time / total_time) * 100

    print(f"Total runtime: {round(t6 - t1, 4)}s")

    print("=" * 60)
    print(f"\nSINGLE-THREADED PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total execution time: {total_time:.4f}s")
    print(f"Total CPU time: {cpu_time:.4f}s")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"Peak memory: {end_memory:.2f} MB")
    print(f"Overall CPU efficiency: {efficiency:.1f}%")

    # Store performance summary
    performance_data = {
        'type': 'single-threaded',
        'total_time': total_time,
        'cpu_time': cpu_time,
        'memory_used_mb': memory_used,
        'peak_memory_mb': end_memory,
        'efficiency_percent': efficiency,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    os.makedirs('results', exist_ok=True)
    with open('results/single_threaded_results.txt', 'a') as f:
        f.write(f"\nSINGLE-THREADED PERFORMANCE SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total execution time: {total_time:.4f}s\n")
        f.write(f"Total CPU time: {cpu_time:.4f}s\n")
        f.write(f"Memory used: {memory_used:.2f} MB\n")
        f.write(f"Peak memory: {end_memory:.2f} MB\n")
        f.write(f"Overall CPU efficiency: {efficiency:.1f}%\n")
        f.write(f"Timestamp: {performance_data['timestamp']}\n")

# def clear_caches():
#     gc.collect()
#     modules_to_remove = [m for m in sys.modules if m.startswith('src.')]
#     for module in modules_to_remove:
#         del sys.modules[module]

if __name__ == "__main__":
    # clear_caches()
    run_main()