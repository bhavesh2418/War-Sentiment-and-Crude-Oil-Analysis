"""
sentiment.py — Dual sentiment analysis: VADER (rule-based) + CardiffNLP RoBERTa (transformer).
"""

import os
import pandas as pd
import numpy as np
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import (
    VADER_POS_THRESHOLD, VADER_NEG_THRESHOLD,
    ROBERTA_MODEL_NAME, ROBERTA_MAX_LENGTH, ROBERTA_BATCH_SIZE,
    ROBERTA_LOCAL_PATH, SENTIMENT_LABELS,
    SENTIMENT_RESULTS_FILE,
)


# ── VADER ─────────────────────────────────────────────────────────────────────

def apply_vader(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Apply VADER sentiment to each row.
    Adds: vader_compound, vader_pos, vader_neu, vader_neg, vader_label
    """
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()

    scores = df[text_col].fillna("").apply(analyzer.polarity_scores)
    df["vader_compound"] = scores.apply(lambda s: s["compound"])
    df["vader_pos"]      = scores.apply(lambda s: s["pos"])
    df["vader_neu"]      = scores.apply(lambda s: s["neu"])
    df["vader_neg"]      = scores.apply(lambda s: s["neg"])

    def _label(c):
        if c >= VADER_POS_THRESHOLD:
            return "Positive"
        elif c <= VADER_NEG_THRESHOLD:
            return "Negative"
        return "Neutral"

    df["vader_label"] = df["vader_compound"].apply(_label)

    print("[VADER] Distribution:")
    print(df["vader_label"].value_counts().to_string())
    return df


# ── RoBERTa ───────────────────────────────────────────────────────────────────

def _load_roberta():
    """Load or download the CardiffNLP Twitter-RoBERTa model."""
    local = ROBERTA_LOCAL_PATH
    if os.path.isdir(local) and os.listdir(local):
        print(f"[RoBERTa] Loading from local cache: {local}")
        tokenizer = AutoTokenizer.from_pretrained(local)
        model     = AutoModelForSequenceClassification.from_pretrained(local)
    else:
        print(f"[RoBERTa] Downloading {ROBERTA_MODEL_NAME} ...")
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
        model     = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME)
        tokenizer.save_pretrained(local)
        model.save_pretrained(local)
        print(f"[RoBERTa] Saved to {local}")
    return tokenizer, model


def apply_roberta(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Apply CardiffNLP Twitter-RoBERTa sentiment.
    Pre-trained on 58M tweets — captures contextual tone in news headlines.
    Adds: roberta_neg, roberta_neu, roberta_pos, roberta_compound, roberta_label
    """
    tokenizer, model = _load_roberta()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    model.eval()

    texts  = df[text_col].fillna("").tolist()
    n      = len(texts)
    batch  = ROBERTA_BATCH_SIZE

    all_probs = []
    for i in range(0, n, batch):
        chunk = texts[i: i + batch]
        enc   = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=ROBERTA_MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        if (i // batch) % 10 == 0:
            print(f"  RoBERTa batch {i // batch + 1}/{-(-n // batch)}")

    probs_arr = np.vstack(all_probs)       # shape: (n, 3) — [neg, neu, pos]
    df = df.copy()
    df["roberta_neg"]      = probs_arr[:, 0]
    df["roberta_neu"]      = probs_arr[:, 1]
    df["roberta_pos"]      = probs_arr[:, 2]
    df["roberta_compound"] = probs_arr[:, 2] - probs_arr[:, 0]   # pos - neg
    df["roberta_label"]    = [
        SENTIMENT_LABELS[np.argmax(p)] for p in probs_arr
    ]

    print("[RoBERTa] Distribution:")
    print(df["roberta_label"].value_counts().to_string())
    return df


# ── Daily aggregation ─────────────────────────────────────────────────────────

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores by date.
    Returns a daily summary DataFrame suitable for correlation with oil prices.
    """
    df["date"] = pd.to_datetime(df["date"])

    agg_cols = {
        "vader_compound":   "mean",
        "vader_label":      lambda x: x.value_counts().idxmax(),
        "roberta_compound": "mean",
        "roberta_label":    lambda x: x.value_counts().idxmax(),
        "title":            "count",
    }
    # Only include roberta columns if they exist
    existing_agg = {k: v for k, v in agg_cols.items() if k in df.columns}

    daily = df.groupby("date").agg(existing_agg).reset_index()
    daily = daily.rename(columns={"title": "article_count"})
    daily = daily.sort_values("date").reset_index(drop=True)

    print(f"[Aggregation] Daily sentiment: {len(daily)} days")
    return daily


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_full_sentiment(df: pd.DataFrame, skip_roberta: bool = False,
                       save: bool = True) -> pd.DataFrame:
    """
    Run VADER, optionally RoBERTa, then save results.
    """
    print("\n[Sentiment] Running VADER...")
    df = apply_vader(df)

    if not skip_roberta:
        print("\n[Sentiment] Running RoBERTa...")
        df = apply_roberta(df)

    if save:
        df.to_csv(SENTIMENT_RESULTS_FILE, index=False)
        print(f"\n  Saved: {SENTIMENT_RESULTS_FILE}  ({len(df)} rows)")

    return df
