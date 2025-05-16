from sklearn.feature_extraction.text import CountVectorizer
import warnings
from tqdm import tqdm

class Countvectorizer:
    def __init__(self):
        print("vectorizing..")
        warnings.simplefilter('ignore')

    def dummy(self, doc):
        return doc

    def count_vectorizer(self, tokens):
        vectorizer = CountVectorizer(
            tokenizer=self.dummy,
            preprocessor=self.dummy
        )

        vectorizer.fit(tokens)
        vectors = vectorizer.transform(tqdm((tokens), total=len(tokens)))

        return vectors