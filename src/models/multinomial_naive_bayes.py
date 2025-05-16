import os
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import shuffle
from tqdm import tqdm
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
processed_data, labels = shuffle(processed_data, labels)

vectorizer = Countvectorizer()
features = vectorizer.count_vectorizer(processed_data, labels)

mnb = MultinomialNB()
mnb.fit(features, labels)
score = mnb.score(features, labels)
print("Naive Bayes Score: ", score)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(mnb, features, labels, cv=cv)

print("CV Scores:", scores)
print(f"Mean Score: {scores.mean()}:.4f")

# If you want to save the model

# with open('model/multinomial_naive_bayes.pkl', 'wb') as file:
# pickle.dump(mnb)
