import concurrent.futures
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer

class TfIdf:
    def __init__(self):
        print("Vectorizing...")

    def vectorize(self, tokenized_docs):
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None
        )

        joined_docs = [' '.join(doc) for doc in tokenized_docs]
        vectorizer.fit(joined_docs)

        def transform_single(self, doc):
            return vectorizer.transform([' '.join(doc)])

        with concurrent.futures.ThreadPoolExecutor() as exe:
            futures = [exe.submit(transform_single, doc) for doc in tokenized_docs]
            vectors = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures))]

        return vstack(vectors)