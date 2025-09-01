import string
import time
import concurrent.futures
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class Preprocess:
    r"""Preprocesses raw text input and returns the cleaned, tokenized output.

    The preprocessing pipeline includes:
        - lowercasing
        - punctuation removal
        - tokenization
        - stopword removal
        - lemmatization

    **methods**:

    - `download_resources()`: Downloads the necessary NLTK resources required for preprocessing.
    - `to_lower(text: str) -> str`: Converts all characters in the input string to lowercase.
    - `remove_punkt(text: str) -> str`: Removes HTML break tags and punctuation from the input string.
    - `tokenize(text: str) -> list[str]`: Splits the input string into individual tokens (words).
    - `remove_stopwords(tokens: list[str]) -> list[str]`: Filters out common English stopwords.
    - `lemmatize(tokens: list[str]) -> list[str]`: Lemmatizes each token to its base form.
    - `preprocess_pipeline(text: str) -> list[str]`: Runs the full preprocessing pipeline on a single string.
    - `process_data(texts: list[str]) -> list[list[str]]`: Processes a list of raw texts in parallel and returns the preprocessed output.
    - `process_data(texts: list[str]) -> list[list[str]]`: Processes a list of raw text strings using multithreading. Returns a list of tokenized, cleaned texts.
    """

    def __init__(self):
        print("\npreprocessing data...")

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
        tokens = self.lemmatize(tokens)
        return tokens

    def process_data(self, texts: list[str]) -> list[list[str]]:
        self.t1 = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor() as exe:
            preprocessed_text = list(
                tqdm((exe.submit(self.preprocess_pipeline, text).result() for text in texts), total=len(texts))
            )

        self.t2 = time.perf_counter()
        print(f"Finished in {round(self.t2 - self.t1, 4)} second(s).")

        return preprocessed_text
