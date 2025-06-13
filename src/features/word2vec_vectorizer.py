from gensim.models import Word2Vec

<<<<<<< HEAD
def Word2vec_Vectorizer(corpus):
    model = Word2Vec(sentences=corpus, vector_size=300, window=5, sg=0, min_count=1, workers=4)
    return model
=======
class word2vecVectorizer():
    def vectorize(self, corpus):
        model = Word2Vec(sentences=corpus, vector_size=300, window=5, sg=0, min_count=1, workers=4)
        return model
>>>>>>> experiment-oops
