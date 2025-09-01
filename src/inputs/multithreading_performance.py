import os
import gc
import sys
import pickle
import time
import psutil
from definitions import ROOT_DIR
from src.inputs.load import Load
from src.inputs.preprocess import Preprocess

def run_main():
    """Demonstration pipeline for loading, preprocessing, and saving data.
    Measures performace metrics and stores it for comparision."""

    os.chdir(ROOT_DIR)

    process = psutil.Process()
    start_total = time.perf_counter()
    start_cpu_total = time.process_time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    load = Load()
    raw_pos, pos_labels = load.load_data('data/raw/train/pos')
    raw_neg, neg_labels = load.load_data('data/raw/train/neg')

    preprocess = Preprocess()
    preprocess.download_resources()

    processed_pos = preprocess.process_data(raw_pos)
    processed_neg = preprocess.process_data(raw_neg)

    os.makedirs('data/processed/train/tokens', exist_ok=True)
    os.makedirs('data/processed/train/labels', exist_ok=True)

    with open("data/processed/train/tokens/pos_reviews_tokens.pkl", "wb") as f:
        pickle.dump(processed_pos, f)

    with open("data/processed/train/labels/pos_labels.pkl", "wb") as f:
        pickle.dump(pos_labels, f)

    with open("data/processed/train/tokens/neg_reviews_tokens.pkl", "wb") as f:
        pickle.dump(processed_neg, f)

    with open("data/processed/train/labels/neg_labels.pkl", "wb") as f:
        pickle.dump(neg_labels, f)

    # Add memory tracking at the end
    end_total = time.perf_counter()
    end_cpu_total = time.process_time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB

    total_time = end_total - start_total
    cpu_time = end_cpu_total - start_cpu_total
    memory_used = end_memory - start_memory
    efficiency = (cpu_time / total_time) * 100

    print("=" * 60)
    print(f"\nMULTITHREADED PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total execution time: {total_time:.4f}s")
    print(f"Total CPU time: {cpu_time:.4f}s")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"Peak memory: {end_memory:.2f} MB")
    print(f"Overall CPU efficiency: {efficiency:.1f}%")

    os.makedirs('results', exist_ok=True)
    with open('results/multithreaded_results.txt', 'a') as f:
        f.write(f"\nMULTITHREADED PERFORMANCE SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total execution time: {total_time:.4f}s\n")
        f.write(f"Total CPU time: {cpu_time:.4f}s\n")
        f.write(f"Memory used: {memory_used:.2f} MB\n")
        f.write(f"Peak memory: {end_memory:.2f} MB\n")
        f.write(f"Overall CPU efficiency: {efficiency:.1f}%\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# def clear_caches():
#     gc.collect()
#     modules_to_remove = [m for m in sys.modules if m.startswith('src.')]
#     for module in modules_to_remove:
#         del sys.modules[module]

if __name__ == "__main__":
    # clear_caches()
    run_main()