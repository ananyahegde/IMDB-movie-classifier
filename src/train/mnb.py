import os
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import shuffle
from tqdm import tqdm
from definitions import ROOT_DIR
from src.inputs.load import Load
from src.inputs.preprocess import Preprocess
from src.features.count_vectorizer import countVectorizer
from src.features.label_encoder import labelEncoder

os.chdir(ROOT_DIR)

path_to_train_pos_data = 'data/raw/train/pos'
path_to_train_neg_data = 'data/raw/train/neg'

print("____________Training the model____________")

load = Load()
raw_train_pos, train_pos_labels = load.load_data(path_to_train_pos_data)
raw_train_neg, train_neg_labels = load.load_data(path_to_train_neg_data)

labels = train_pos_labels + train_neg_labels

encoder = labelEncoder()
labels = [encoder.map_label(label) for label in labels]

preprocess = Preprocess()
processed_train_pos = preprocess.process_data(raw_train_pos)
processed_train_neg = preprocess.process_data(raw_train_neg)

processed_train_data = processed_train_pos + processed_train_neg
processed_train_data, labels = shuffle(processed_train_data, labels)

vectorizer = countVectorizer()
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

path_to_train_pos_data = 'data/raw/test/pos'
path_to_train_neg_data = 'data/raw/test/neg'

print("____________Testing the model____________")

load = Load()
raw_test_pos, test_pos_labels = load.load_data(path_to_test_pos_data)
raw_test_neg, test_neg_labels = load.load_data(path_to_test_neg_data)

test_labels = test_pos_labels + test_neg_labels

encoder = labelEncoder()
test_labels = [encoder.map_label(label) for label in test_labels]

preprocess = Preprocess()
processed_test_pos = preprocess.process_data(raw_test_pos)
processed_test_neg = preprocess.process_data(raw_test_neg)

processed_test_data = processed_test_pos + processed_test_neg
processed_test_data, labels = shuffle(processed_test_data, labels)
