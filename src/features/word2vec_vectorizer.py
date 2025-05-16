from gensim.models import Word2Vec

class Word2vec:
    def __init__(self):
        print("Vectorizing...")

    def Word2vec_Vectorizer(self, corpus):
        model = Word2Vec(sentences=corpus, vector_size=300, window=5, sg=0, min_count=1, workers=4)
        return model
