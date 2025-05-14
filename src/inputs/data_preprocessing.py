import os
import pickle
import time
from definitions import ROOT_DIR
from .load import load_data
from .preprocess import process_data
from .preprocess import download_resources

def run_main():
    # setting the working directory
    os.chdir(ROOT_DIR)

    # Load data
    t1 = time.perf_counter()

    print('loading and importing data...')
    raw_pos, pos_labels = load_data('data/raw/train/pos')
    raw_neg, neg_labels = load_data('data/raw/train/neg')

    t2 = time.perf_counter()
    print(f"Loaded {len(raw_pos)} positive and {len(raw_neg)} negative reviews in {round(t2 - t1, 4)} second(s).")

    # download necessery resources
    download_resources()

    # preprocessing data
    t3 = time.perf_counter()

    print('preprocessing data...')
    processed_pos = process_data(raw_pos)
    processed_neg = process_data(raw_neg)

    t4 = time.perf_counter()
    print(f"data preprocessing finished in {round(t4 - t3, 4)} second(s).")

    # save the processed data
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