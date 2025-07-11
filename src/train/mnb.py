import os
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tqdm import tqdm
from definitions import ROOT_DIR
from src.inputs.load import Load
from src.inputs.preprocess import Preprocess
from src.features.count_vectorizer import countVectorizer
from src.features.binary_mapping import labelEncoder


os.chdir(ROOT_DIR)

path_to_train_pos_data = 'data/raw/train/pos'
path_to_train_neg_data = 'data/raw/train/neg'

print("____________Training the model____________")

if not (os.path.exists('data/interim/mnb_features.pkl') and os.path.getsize('data/interim/mnb_features.pkl') > 0) \
        or not (os.path.exists('data/interim/mnb_labels.pkl') and os.path.getsize('data/interim/mnb_labels.pkl') > 0) \
        or not (os.path.exists('models/best_count_vectorizer.pkl') and os.path.getsize('models/best_count_vectorizer.pkl') > 0):

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
    best_count_vectorizer, features = vectorizer.vectorize(processed_train_data, labels)

    with open('data/interim/mnb_features.pkl', 'wb') as file:
        pickle.dump(features, file)

    with open('data/interim/mnb_labels.pkl', 'wb') as file:
        pickle.dump(labels, file)

    with open('models/best_count_vectorizer.pkl', 'wb') as file:
        pickle.dump(best_count_vectorizer, file)

else:
    with open('data/interim/mnb_features.pkl', 'rb') as file:
        features = pickle.load(file)

    with open('data/interim/mnb_labels.pkl', 'rb') as file:
        labels = pickle.load(file)


mnb = MultinomialNB()
mnb.fit(features, labels)

score = mnb.score(features, labels)
print("Naive Bayes Score:", score)

train_preds = mnb.predict(features)
train_cm = confusion_matrix(labels, train_preds)
print(f"Confusion matrix for train predictions:\n {train_cm}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(mnb, features, labels, cv=cv)
print("CV Scores:", scores)
print(f"Mean Score: {scores.mean():.4f}")

with open('models/multinomial_naive_bayes.pkl', 'wb') as file:
    pickle.dump(mnb, file)


path_to_test_pos_data = 'data/raw/test/pos'
path_to_test_neg_data = 'data/raw/test/neg'

if not (os.path.exists('data/interim/mnb_test_features.pkl') and os.path.getsize('data/interim/mnb_test_features.pkl') > 0) \
        or not (os.path.exists('data/interim/mnb_test_labels.pkl') and os.path.getsize('data/interim/mnb_test_labels.pkl') > 0):

    with open('models/best_count_vectorizer.pkl', 'rb') as file:
        best_count_vectorizer = pickle.load(file)

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
    processed_test_data, test_labels = shuffle(processed_test_data, test_labels)

    test_features = best_count_vectorizer.transform(processed_test_data)

    with open('data/interim/mnb_test_features.pkl', 'wb') as file:
        pickle.dump(test_features, file)
    with open('data/interim/mnb_test_labels.pkl', 'wb') as file:
        pickle.dump(test_labels, file)

else:
    with open('data/interim/mnb_test_features.pkl', 'rb') as file:
        test_features = pickle.load(file)

    with open('data/interim/mnb_test_labels.pkl', 'rb') as file:
        test_labels = pickle.load(file)

    with open('models/multinomial_naive_bayes.pkl', 'rb') as file:
        mnb = pickle.load(file)

mnb_test_score = mnb.score(test_features, test_labels)
print(f"Naive Bayes test score: {mnb_test_score}")

test_preds = mnb.predict(test_features)
test_cm = confusion_matrix(test_labels, test_preds)
print(f"Confusion matrix for test predictions:\n {test_cm}")
