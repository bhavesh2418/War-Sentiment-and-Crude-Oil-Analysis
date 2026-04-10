"""
data_collector.py — Collect news articles via Google News RSS + NewsAPI
and crude oil prices via yfinance. All timestamped.
"""

import pandas as pd
import feedparser
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time
import re

from src.config import (
    NEWS_API_KEY, SEARCH_KEYWORDS, GNEWS_RSS_URL, NEWSAPI_URL,
    COLLECTION_DAYS, OIL_TICKERS,
    NEWS_RAW_FILE, CRUDE_RAW_FILE,
)


# ── Google News RSS ───────────────────────────────────────────────────────────

def collect_google_news_rss(keywords=None, max_per_keyword=50) -> pd.DataFrame:
    """Collect news from Google News RSS for each keyword."""
    if keywords is None:
        keywords = SEARCH_KEYWORDS

    records = []
    for kw in keywords:
        query = kw.replace(" ", "+")
        url = GNEWS_RSS_URL.format(query=query)
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_keyword]:
                pub = entry.get("published", "")
                try:
                    pub_dt = pd.to_datetime(pub, utc=True)
                except Exception:
                    pub_dt = pd.NaT

                summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))
                records.append({
                    "source":    "Google News RSS",
                    "keyword":   kw,
                    "title":     entry.get("title", ""),
                    "text":      summary,
                    "url":       entry.get("link", ""),
                    "published": pub_dt,
                })
            print(f"  RSS [{kw}]: {len(feed.entries[:max_per_keyword])} articles")
            time.sleep(0.5)
        except Exception as e:
            print(f"  RSS [{kw}] ERROR: {e}")

    return pd.DataFrame(records)


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def collect_newsapi(keywords=None, days_back=None, page_size=100) -> pd.DataFrame:
    """Collect news from NewsAPI for each keyword."""
    if not NEWS_API_KEY:
        print("NEWS_API_KEY not set — skipping NewsAPI collection")
        return pd.DataFrame()

    if keywords is None:
        keywords = SEARCH_KEYWORDS
    if days_back is None:
        days_back = COLLECTION_DAYS

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    records = []

    for kw in keywords:
        params = {
            "q":        kw,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": page_size,
            "from":     from_date,
            "apiKey":   NEWS_API_KEY,
        }
        try:
            resp = requests.get(NEWSAPI_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for art in data.get("articles", []):
                text = " ".join(filter(None, [
                    art.get("title", ""),
                    art.get("description", ""),
                ]))
                records.append({
                    "source":    art.get("source", {}).get("name", "NewsAPI"),
                    "keyword":   kw,
                    "title":     art.get("title", ""),
                    "text":      text,
                    "url":       art.get("url", ""),
                    "published": pd.to_datetime(
                        art.get("publishedAt"), utc=True, errors="coerce"
                    ),
                })
            count = len(data.get("articles", []))
            print(f"  NewsAPI [{kw}]: {count} articles  "
                  f"(total available: {data.get('totalResults', '?')})")
            time.sleep(0.3)
        except Exception as e:
            print(f"  NewsAPI [{kw}] ERROR: {e}")

    return pd.DataFrame(records)


# ── Combine + Deduplicate ─────────────────────────────────────────────────────

def collect_all_news(save=True) -> pd.DataFrame:
    """Collect from all sources, deduplicate, and save."""
    print("\n[Data Collection] Google News RSS...")
    rss_df = collect_google_news_rss()

    print("\n[Data Collection] NewsAPI...")
    api_df = collect_newsapi()

    df = pd.concat([rss_df, api_df], ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset=["title"])
    df = df[df["title"].str.strip() != ""]
    df = df.dropna(subset=["title"])
    print(f"\n  Combined: {before} -> {len(df)} after deduplication")

    df = df.sort_values("published", ascending=False).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["published"]).dt.date

    if save:
        df.to_csv(NEWS_RAW_FILE, index=False)
        print(f"  Saved: {NEWS_RAW_FILE}  ({len(df)} rows)")

    return df


# ── Crude Oil Prices ──────────────────────────────────────────────────────────

def collect_crude_oil(days_back=None, save=True) -> pd.DataFrame:
    """Download Brent and WTI crude oil daily prices via yfinance."""
    if days_back is None:
        days_back = COLLECTION_DAYS + 30  # extra buffer for merge

    end   = datetime.utcnow()
    start = end - timedelta(days=days_back)

    frames = []
    for name, ticker in OIL_TICKERS.items():
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
            )
            df = df[["Close"]].rename(columns={"Close": f"{name}_Close"})
            df.index = pd.to_datetime(df.index).date
            frames.append(df)
            print(f"  yfinance [{name} {ticker}]: {len(df)} trading days")
        except Exception as e:
            print(f"  yfinance [{name}] ERROR: {e}")

    if not frames:
        return pd.DataFrame()

    crude = pd.concat(frames, axis=1)

    # Flatten any multi-level column index from yfinance
    if isinstance(crude.columns, pd.MultiIndex):
        crude.columns = [c[0] for c in crude.columns]

    crude = crude.reset_index()
    crude.columns = [str(c) for c in crude.columns]
    crude = crude.rename(columns={crude.columns[0]: "date"})

    for col in ["Brent_Close", "WTI_Close"]:
        if col in crude.columns:
            crude[col.replace("Close", "Return")] = crude[col].pct_change() * 100

    if save:
        crude.to_csv(CRUDE_RAW_FILE, index=False)
        print(f"  Saved: {CRUDE_RAW_FILE}  ({len(crude)} rows)")

    return crude


if __name__ == "__main__":
    news  = collect_all_news()
    crude = collect_crude_oil()
    print(f"\nNews shape : {news.shape}")
    print(f"Crude shape: {crude.shape}")
