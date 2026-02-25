"""NLTK-based text preprocessing for the Study Buddy Bot."""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def initialize():
    """Download required NLTK data (call once at startup)."""
    for resource in ["punkt_tab", "stopwords", "wordnet"]:
        nltk.download(resource, quiet=True)


_lemmatizer = WordNetLemmatizer()
_stop_words = None


def _get_stop_words() -> set:
    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words("english"))
    return _stop_words


def tokenize(text: str) -> list[str]:
    """Return NLTK word tokens (lowercase)."""
    return word_tokenize(text.lower())


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter out English stopwords."""
    sw = _get_stop_words()
    return [t for t in tokens if t not in sw]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize each token using WordNetLemmatizer."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str) -> str:
    """Full preprocessing pipeline for intent classification input.

    Lowercase -> tokenize -> remove stopwords -> lemmatize -> rejoin.
    """
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def clean_for_embedding(text: str) -> str:
    """Light cleaning for sentence-transformer input (preserve natural text)."""
    return text.strip()
