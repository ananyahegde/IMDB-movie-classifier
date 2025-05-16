import os
import pickle
from sklearn.naive_bayes import MultinomialNB
from definitions import ROOT_DIR
from src.inputs.load import Load
from src.inputs.preprocess import Preprocess
from src.features.count_vectorizer import Countvectorizer
os.chdir(ROOT_DIR)

path_to_pos_data = 'data/raw/train/pos'
path_to_neg_data = 'data/raw/train/neg'
load = Load()
raw_pos, pos_labels = load.load_data(path_to_pos_data)
raw_neg, neg_labels = load.load_data(path_to_neg_data)

labels = pos_labels + neg_labels

preprocess = Preprocess()
processed_pos = preprocess.process_data(raw_pos)
processed_neg = preprocess.process_data(raw_neg)

processed_data = processed_pos + processed_neg

vectorizer = Countvectorizer()
features = vectorizer.count_vectorizer(processed_data)


# with open("data/processed/train/embeddings/count_vectors.pkl", "rb") as file:
#     features = pickle.load(file)
#
# with open("data/processed/train/labels/labels.pkl", "rb") as file:
#     labels = pickle.load(file)
#
# mnb = MultinomialNB()
# mnb.fit(features, labels)
# score = mnb.score(features, labels)
# print(score)
#
# from sklearn.model_selection import cross_val_score
#
# scores = cross_val_score(mnb, features, labels, cv=5)
# print(scores)
#
# with open('model/multinomial_naive_bayes.pkl', 'wb') as file:
#     pickle.dump(mnb)
#
#
