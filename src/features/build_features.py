import os
import pickle
import time
from definitions import ROOT_DIR
import numpy as np
from scipy.sparse import vstack
from .count_vectorizer import Count_Vectorizer
from .tfidf_vectorizer import Tfidf_Vectorizer
from .word2vec_vectorizer import Word2vec_Vectorizer

def run_main():
    os.chdir(ROOT_DIR)

    with open('data/processed/train/tokens/pos_reviews_tokens.pkl', 'rb') as file:
        pos_reviews = pickle.load(file)

    with open('data/processed/train/labels/pos_labels.pkl', 'rb') as file:
        pos_labels = pickle.load(file)

    with open('data/processed/train/tokens/neg_reviews_tokens.pkl', 'rb') as file:
        neg_reviews = pickle.load(file)

    with open('data/processed/train/labels/neg_labels.pkl', 'rb') as file:
        neg_labels = pickle.load(file)

    print("loaded the processed data.")

    reviews_tokens = pos_reviews + neg_reviews
    labels = pos_labels + neg_labels
    labels = np.where(np.array(labels) < 5, 0, 1)

    print("Vectorizing...")
    t1 = time.perf_counter()

    count_vectors = Count_Vectorizer(reviews_tokens)
    tfidf_vectors = Tfidf_Vectorizer(reviews_tokens)
    word2vec_embeddings = Word2vec_Vectorizer(reviews_tokens)

    t2 = time.perf_counter()
    print(f"Vectorization finished in {round(t2 - t1, 4)} second(s).")


    os.makedirs('data/processed/train/embeddings', exist_ok=True)

    with open("data/processed/train/embeddings/count_vectors.pkl", "wb") as f:
        pickle.dump(count_vectors, f)

    with open("data/processed/train/embeddings/tfidf_vectors.pkl", "wb") as f:
        pickle.dump(tfidf_vectors, f)

    with open("data/processed/train/embeddings/word2vec_embeddings.pkl", "wb") as f:
        pickle.dump(word2vec_embeddings, f)

    with open("data/processed/train/labels/labels.pkl", "wb") as f:
        pickle.dump(labels, f)


