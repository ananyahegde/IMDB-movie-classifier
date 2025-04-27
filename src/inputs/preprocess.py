import string
import concurrent.futures
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def download_resources():
    print("downloading resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('finished downloading resources.')

# Lowercase correction
def to_lower(text: str) -> str:
    return text.lower()

# Punctuation removal correction
def remove_punkt(text: str) -> str:
    text = text.replace("<br />", "")
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

# Tokenization
def tokenize(text: str) -> list[str]:
    return word_tokenize(text)

# Lemmatization
def lemmatize(tokens: list[str]) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Stopword removal
def remove_stopwords(tokens: list[str]) -> list[str]:
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

# Preprocessing Pipeline
def preprocess_pipeline(text: str) -> list[str]:
    text = to_lower(text)
    text = remove_punkt(text)
    tokens = tokenize(text)
    tokens =  remove_stopwords(tokens)
    return lemmatize(tokens)


def process_data(texts: list[str]) -> list[list[str]]:
    with concurrent.futures.ThreadPoolExecutor() as exe:
        preprocessed_text = [exe.submit(preprocess_pipeline, text).result() for text in texts]
        return preprocessed_text
