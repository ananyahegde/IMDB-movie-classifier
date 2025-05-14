import os
import pickle
import time
import numpy as np
from scipy.sparse import vstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from definitions import ROOT_DIR
from src.inputs.load import load_data
from src.inputs.preprocess import process_data
from src.features.count_vectorizer import Count_Vectorizer

os.chdir(ROOT_DIR)

t1 = time.perf_counter()

print('loading and importing data...')
test_raw_pos, test_pos_labels = load_data('data/raw/test/pos')
test_raw_neg, test_neg_labels = load_data('data/raw/test/neg')

t2 = time.perf_counter()
print(f"Loaded {len(test_raw_pos)} positive and {len(test_raw_neg)} negative reviews in {round(t2 - t1, 4)} second(s).")

# preprocessing data
t3 = time.perf_counter()

print('preprocessing data...')
test_processed_pos = process_data(test_raw_pos)
test_processed_neg = process_data(test_raw_neg)

t4 = time.perf_counter()
print(f"data preprocessing finished in {round(t4 - t3, 4)} second(s).")

test_tokens = test_processed_pos + test_processed_neg
labels = test_pos_labels + test_neg_labels
labels = np.where(np.array(labels) < 5, 0, 1)

print("Vectorizing...")
t1 = time.perf_counter()

features = Count_Vectorizer(reviews_tokens)

t2 = time.perf_counter()
print(f"Vectorization finished in {round(t2 - t1, 4)} second(s).")

with open('model/multinomial_naive_bayes.pkl', 'rb') as f:
    model = pickle.load(f)

score = model.score(features, labels)
print(score)

scores = cross_val_score(mnb, features, labels, cv=5)
print(scores)
