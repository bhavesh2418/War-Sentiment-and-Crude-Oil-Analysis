"""
main.py — Full pipeline runner for War Sentiment & Crude Oil Analysis.

Stages:
  1. Data Collection  — Google News RSS + NewsAPI + yfinance
  2. Preprocessing    — Text cleaning, tokenization, lemmatization
  3. EDA Plots        — Article trends, source distribution, word clouds
  4. Sentiment        — VADER + RoBERTa (optional)
  5. Aggregation      — Daily sentiment scores
  6. Correlation      — Merge with oil prices, Pearson r, rolling plots

Usage:
  python main.py                  # full pipeline with RoBERTa
  python main.py --skip-roberta   # VADER only (faster)
  python main.py --use-cache      # skip data collection if raw files exist
"""

import argparse
import sys
import os
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    NEWS_RAW_FILE, CRUDE_RAW_FILE, NEWS_PROCESSED_FILE,
    SENTIMENT_RESULTS_FILE, CORRELATION_FILE,
)
from src.data_collector import collect_all_news, collect_crude_oil
from src.preprocessing  import preprocess_news
from src.sentiment      import run_full_sentiment, aggregate_daily
from src.visualize      import (
    plot_crude_oil_trend, plot_articles_over_time, plot_source_distribution,
    plot_keyword_distribution, plot_token_length_distribution,
    plot_wordcloud, plot_wordcloud_all,
    plot_sentiment_distribution, plot_sentiment_over_time,
    plot_sentiment_vs_oil, plot_correlation_scatter,
    plot_rolling_correlation, plot_model_agreement,
)


def parse_args():
    p = argparse.ArgumentParser(description="War Sentiment & Crude Oil Pipeline")
    p.add_argument("--skip-roberta", action="store_true",
                   help="Skip RoBERTa transformer (use VADER only)")
    p.add_argument("--use-cache", action="store_true",
                   help="Use existing raw data files if present")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print(" War Sentiment & Crude Oil Analysis — Full Pipeline")
    print("=" * 65)

    # ── Stage 1: Data Collection ──────────────────────────────────────────
    print("\n[Stage 1] Data Collection")
    if args.use_cache and NEWS_RAW_FILE.exists() and CRUDE_RAW_FILE.exists():
        print("  Using cached files.")
        news_df  = pd.read_csv(NEWS_RAW_FILE)
        crude_df = pd.read_csv(CRUDE_RAW_FILE)
    else:
        news_df  = collect_all_news(save=True)
        crude_df = collect_crude_oil(save=True)

    print(f"  News : {news_df.shape}  |  Crude: {crude_df.shape}")

    # ── Stage 2: Preprocessing ────────────────────────────────────────────
    print("\n[Stage 2] Preprocessing")
    proc_df = preprocess_news(news_df, save=True)
    print(f"  Processed: {proc_df.shape}")

    # ── Stage 3: EDA Plots ────────────────────────────────────────────────
    print("\n[Stage 3] EDA Visualizations")
    plot_crude_oil_trend(crude_df)
    plot_articles_over_time(proc_df)
    plot_source_distribution(proc_df)
    plot_keyword_distribution(proc_df)
    plot_token_length_distribution(proc_df)
    plot_wordcloud_all(proc_df)

    # ── Stage 4: Sentiment Analysis ───────────────────────────────────────
    print("\n[Stage 4] Sentiment Analysis")
    sent_df = run_full_sentiment(proc_df, skip_roberta=args.skip_roberta, save=True)

    # ── EDA plots that need sentiment labels ──────────────────────────────
    plot_wordcloud(sent_df)
    plot_sentiment_distribution(sent_df)

    # ── Stage 5: Daily Aggregation ────────────────────────────────────────
    print("\n[Stage 5] Daily Aggregation")
    daily_df = aggregate_daily(sent_df)
    plot_sentiment_over_time(daily_df)

    # ── Stage 6: Correlation Analysis ────────────────────────────────────
    print("\n[Stage 6] Correlation Analysis")
    crude_df["date"] = pd.to_datetime(crude_df["date"]).dt.date
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date

    merged = pd.merge(daily_df, crude_df, on="date", how="inner")
    print(f"  Merged: {len(merged)} days with both sentiment and oil data")

    # Pearson correlations
    results = []
    for sent_col, sent_name in [("vader_compound", "VADER"),
                                  ("roberta_compound", "RoBERTa")]:
        if sent_col not in merged.columns:
            continue
        for price_col, market in [("Brent_Close", "Brent"), ("WTI_Close", "WTI")]:
            if price_col not in merged.columns:
                continue
            sub = merged[[sent_col, price_col]].dropna()
            r   = sub[sent_col].corr(sub[price_col])
            strength = (
                "Strong" if abs(r) >= 0.5 else
                "Moderate" if abs(r) >= 0.3 else
                "Weak"
            )
            direction = "positive" if r > 0 else "negative"
            results.append({
                "Model": sent_name, "Market": market,
                "Pearson_r": round(r, 4),
                "Interpretation": f"{strength} {direction} correlation",
            })
            print(f"  {sent_name} vs {market}: r = {r:.4f}  ({strength} {direction})")

    corr_df = pd.DataFrame(results)
    corr_df.to_csv(CORRELATION_FILE, index=False)
    print(f"  Saved: {CORRELATION_FILE}")

    # Model agreement
    plot_model_agreement(sent_df)

    # Correlation plots
    plot_sentiment_vs_oil(merged)
    plot_correlation_scatter(merged)
    plot_rolling_correlation(merged)

    # ── Final merged dataset ──────────────────────────────────────────────
    merged.to_csv(SENTIMENT_RESULTS_FILE.parent / "merged_sentiment_oil.csv", index=False)

    print("\n" + "=" * 65)
    print(" Pipeline complete.")
    print(f"  Images  : images/  ({len(list(__import__('src.config', fromlist=['IMAGES_DIR']).IMAGES_DIR.glob('*.png')))} plots)")
    print(f"  Reports : reports/")
    print("=" * 65)


if __name__ == "__main__":
    main()
