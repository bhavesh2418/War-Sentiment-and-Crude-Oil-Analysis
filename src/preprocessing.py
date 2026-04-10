"""
preprocessing.py — Text cleaning pipeline for NLP sentiment analysis on news content.
"""

import re
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config import DOMAIN_STOPWORDS, NEWS_PROCESSED_FILE

# Download required NLTK data on first run
for _pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_tokenizer  = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
_lemmatizer = WordNetLemmatizer()
_stopwords  = stopwords.words("english")
_all_stops  = set(_stopwords) | DOMAIN_STOPWORDS


def clean_text(text: str) -> str:
    """
    Clean a single text string through 8 sequential operations:
    1. Strip URLs
    2. Strip HTML tags
    3. Remove Twitter handles and hashtag symbols
    4. Remove special characters and digits
    5. Tokenize with TweetTokenizer (preserves contractions)
    6. Filter short tokens and stopwords
    7. Lemmatize
    8. Rejoin into clean string
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # 2. HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # 3. Twitter handles and hashtag symbols (keep hashtag word)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    # 4. Special characters and digits
    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    # 5. Tokenize
    tokens = _tokenizer.tokenize(text)

    # 6. Filter short tokens and stopwords
    tokens = [t for t in tokens if len(t) >= 3 and t not in _all_stops]

    # 7. Lemmatize
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_news(df: pd.DataFrame, text_col: str = "text",
                    title_col: str = "title", save: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Combines title + body text
    - Applies clean_text()
    - Filters documents with fewer than 11 characters after cleaning
    - Adds token_count and standardized date column
    """
    df = df.copy()

    # Combine title and body
    df["combined_text"] = (
        df[title_col].fillna("") + " " + df[text_col].fillna("")
    ).str.strip()

    print("[Preprocessing] Cleaning text...")
    df["cleaned_text"] = df["combined_text"].apply(clean_text)

    # Filter out near-empty documents
    before = len(df)
    df = df[df["cleaned_text"].str.len() >= 11].reset_index(drop=True)
    print(f"  Removed {before - len(df)} rows with insufficient cleaned text")

    # Token count
    df["token_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))

    # Standardize date
    df["date"] = pd.to_datetime(df["published"], utc=True, errors="coerce").dt.date

    print(f"  Processed: {len(df)} articles | "
          f"vocab size (approx): {df['cleaned_text'].str.split().explode().nunique()}")

    if save:
        df.to_csv(NEWS_PROCESSED_FILE, index=False)
        print(f"  Saved: {NEWS_PROCESSED_FILE}")

    return df
