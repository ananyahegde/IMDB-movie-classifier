from sklearn.feature_extraction.text import CountVectorizer

def dummy(doc):
    return doc

def Count_Vectorizer(tokens):
    vectorizer = CountVectorizer(
        tokenizer=dummy,
        preprocessor=dummy,
    )

    vectorizer.fit(tokens)
    vectors = vectorizer.transform(tokens)

    return vectors