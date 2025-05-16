import string
import time
import concurrent.futures
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class Preprocess:
    def __init__(self):
        print("preprocessing data...")
    def download_resources(self):
        print("downloading resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        print('finished downloading resources.')

    def to_lower(self, text: str) -> str:
        return text.lower()

    def remove_punkt(self, text: str) -> str:
        text = text.replace("<br />", "")
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')
        return text

    def tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)

    def lemmatize(self, tokens: list[str]) -> list[str]:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        stop_words = set(stopwords.words('english'))
        return [token for token in tokens if token not in stop_words]

    def preprocess_pipeline(self, text: str) -> list[str]:
        text = self.to_lower(text)
        text = self.remove_punkt(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return self.lemmatize(tokens)

    def process_data(self, texts: list[str]) -> list[list[str]]:
        self.t1 = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor() as exe:
            preprocessed_text = list(
                tqdm((exe.submit(self.preprocess_pipeline, text).result() for text in texts), total=len(texts))
            )

        self.t2 = time.perf_counter()
        print(f"Finished in {round(self.t2 - self.t1, 4)} second(s).")

        return preprocessed_text
