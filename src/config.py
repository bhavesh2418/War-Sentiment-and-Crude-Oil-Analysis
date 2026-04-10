"""
config.py — Central configuration for War Sentiment & Crude Oil Analysis.
All paths, constants, and model parameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Directories ───────────────────────────────────────────────────────────────
DATA_RAW_DIR       = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
IMAGES_DIR         = ROOT / "images"
MODELS_DIR         = ROOT / "models"
REPORTS_DIR        = ROOT / "reports"

for _d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, IMAGES_DIR, MODELS_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── File paths ────────────────────────────────────────────────────────────────
NEWS_RAW_FILE          = DATA_RAW_DIR       / "news_raw.csv"
CRUDE_RAW_FILE         = DATA_RAW_DIR       / "crude_oil_prices.csv"
NEWS_PROCESSED_FILE    = DATA_PROCESSED_DIR / "news_processed.csv"
SENTIMENT_RESULTS_FILE = REPORTS_DIR        / "sentiment_model_results.csv"
CORRELATION_FILE       = REPORTS_DIR        / "sentiment_oil_correlation.csv"
PDF_REPORT_PATH        = REPORTS_DIR        / "War_Sentiment_Process_Report.pdf"

# ── Credentials ───────────────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ── Data collection ───────────────────────────────────────────────────────────
COLLECTION_DAYS = 150

SEARCH_KEYWORDS = [
    "US Israel Iran war oil",
    "Iran sanctions oil",
    "Israel Iran conflict crude",
    "strait of hormuz oil",
    "Middle East war oil price",
    "Iran nuclear deal oil",
    "Gaza war oil market",
]

GNEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
NEWSAPI_URL   = "https://newsapi.org/v2/everything"

# ── Oil tickers ───────────────────────────────────────────────────────────────
OIL_TICKERS = {
    "Brent": "BZ=F",
    "WTI":   "CL=F",
}

# ── VADER thresholds ──────────────────────────────────────────────────────────
VADER_POS_THRESHOLD =  0.05
VADER_NEG_THRESHOLD = -0.05

# ── RoBERTa ───────────────────────────────────────────────────────────────────
ROBERTA_MODEL_NAME  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ROBERTA_MAX_LENGTH  = 128
ROBERTA_BATCH_SIZE  = 32
ROBERTA_LOCAL_PATH  = str(MODELS_DIR / "roberta_sentiment")

# ── Sentiment labels ──────────────────────────────────────────────────────────
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

# ── Domain-specific stopwords (common in news but not analytically useful) ────
DOMAIN_STOPWORDS = {
    "reuters", "ap", "official", "said", "say", "says", "told",
    "according", "report", "reporting", "reported", "news", "article",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "year", "years", "week", "weeks", "day", "days", "time", "times",
    "new", "also", "would", "could", "one", "two", "three",
}
