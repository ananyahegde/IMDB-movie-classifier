from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint
import warnings


class countVectorizer:
    r"""Applies sklearn's `CountVectorizer()` to a series of tokens.

    Also performs randomized hyperparameter search using a Naive Bayes classifier
    to find the best vectorizer configuration.

    Returns the fitted CountVectorizer and the corresponding document-term matrix.

    See: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    """

    def __init__(self):
        print("vectorizing..")
        warnings.simplefilter('ignore')

    def dummy(self, doc):
        return doc

    def vectorize(self, tokens, labels):
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
        }

        random_search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=3,
            cv=5,
            scoring='accuracy',
            n_jobs=1,
            verbose=0,
            random_state=42
        )

        random_search.fit(tokens, labels)

        best_vectorizer = random_search.best_estimator_.named_steps['vect']
        vectors = best_vectorizer.transform(tokens)

        return best_vectorizer, vectors
