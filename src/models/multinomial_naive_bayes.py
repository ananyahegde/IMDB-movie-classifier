import os
import pickle
from sklearn.naive_bayes import MultinomialNB

os.chdir('C:/Users/Anany/Wo Maschinen lernen/Text Classification of IMDB Reviews')

with open("data/processed/train/embeddings/count_vectors.pkl", "rb") as file:
    features = pickle.load(file)

with open("data/processed/train/labels/labels.pkl", "rb") as file:
    labels = pickle.load(file)

mnb = MultinomialNB()
mnb.fit(features, labels)
score = mnb.score(features, labels)
print(score)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(mnb, features, labels, cv=5)
print(scores)
