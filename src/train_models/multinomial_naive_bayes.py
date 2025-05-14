import os
import pickle
from definitions import ROOT_DIR
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

os.chdir(ROOT_DIR)

with open("data/processed/train/embeddings/count_vectors.pkl", "rb") as file:
    features = pickle.load(file)

with open("data/processed/train/labels/labels.pkl", "rb") as file:
    labels = pickle.load(file)

mnb = MultinomialNB()
mnb.fit(features, labels)
score = mnb.score(features, labels)
print(score)

scores = cross_val_score(mnb, features, labels, cv=5)
print(scores)

with open('model/multinomial_naive_bayes.pkl', 'wb') as f:
    pickle.dump(mnb, f)