from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint
from tqdm import tqdm
import warnings


class Countvectorizer:
    def __init__(self):
        print("vectorizing..")
        warnings.simplefilter('ignore')

    def dummy(self, doc):
        return doc


    def count_vectorizer(self, tokens, labels):
        pipe = Pipeline([
            ('vect', CountVectorizer(
                tokenizer=self.dummy,
                preprocessor=self.dummy
            )),
            ('clf', MultinomialNB())
        ])

        param_dist = {
            'vect__max_df': uniform(0.7, 0.3),
            'vect__min_df': randint(1, 4),
            'vect__ngram_range': [(1, 1), (1, 2)],
            'clf__alpha': uniform(0.1, 1.0)
        }

        random_search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=5,
            cv=5,
            scoring='accuracy',
            n_jobs=1,
            verbose=2,
            random_state=42
        )

        random_search.fit(tokens, labels)

        best_vectorizer = random_search.best_estimator_.named_steps['vect']
        vectors = best_vectorizer.transform(tokens)

        return vectors
