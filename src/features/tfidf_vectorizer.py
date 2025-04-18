import concurrent.futures
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer

def Tfidf_Vectorizer(tokenized_docs):
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None
    )

    joined_docs = [' '.join(doc) for doc in tokenized_docs]
    vectorizer.fit(joined_docs)

    def transform_single(doc):
        return vectorizer.transform([' '.join(doc)])

    with concurrent.futures.ThreadPoolExecutor() as exe:
        futures = [exe.submit(transform_single, doc) for doc in tokenized_docs]
        vectors = [future.result() for future in concurrent.futures.as_completed(futures)]

    return vstack(vectors)