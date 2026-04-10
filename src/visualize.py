"""
visualize.py — All plot functions. Saves to images/ at 150 DPI.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from pathlib import Path

from src.config import IMAGES_DIR

sns.set_theme(style="whitegrid", palette="Set2")
_COLORS = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#9E9E9E"}


def _save(name: str) -> None:
    path = IMAGES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 00: Crude oil trend ───────────────────────────────────────────────────────

def plot_crude_oil_trend(crude_df: pd.DataFrame) -> None:
    """Line chart: Brent and WTI closing prices over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    crude_df = crude_df.copy()
    crude_df["date"] = pd.to_datetime(crude_df["date"])

    if "Brent_Close" in crude_df.columns:
        ax.plot(crude_df["date"], crude_df["Brent_Close"], label="Brent", linewidth=2)
    if "WTI_Close" in crude_df.columns:
        ax.plot(crude_df["date"], crude_df["WTI_Close"], label="WTI",
                linewidth=2, linestyle="--")

    ax.set_title("Crude Oil Prices — Brent & WTI", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/bbl)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend()
    _save("00_crude_oil_trend.png")


# ── 01: Articles over time ────────────────────────────────────────────────────

def plot_articles_over_time(df: pd.DataFrame) -> None:
    """Bar chart: daily article count."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily_counts = df.groupby("date").size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(daily_counts["date"], daily_counts["count"], color="#5C85D6", width=0.8)
    ax.set_title("News Articles Collected Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Article Count")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    _save("01_articles_over_time.png")


# ── 02: Source distribution ───────────────────────────────────────────────────

def plot_source_distribution(df: pd.DataFrame, top_n: int = 15) -> None:
    """Horizontal bar chart: top news sources."""
    top = df["source"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    top.sort_values().plot(kind="barh", ax=ax, color="#5C85D6")
    ax.set_title(f"Top {top_n} News Sources", fontsize=14, fontweight="bold")
    ax.set_xlabel("Article Count")
    _save("02_source_distribution.png")


# ── 03: Keyword distribution ──────────────────────────────────────────────────

def plot_keyword_distribution(df: pd.DataFrame) -> None:
    """Bar chart: articles per search keyword."""
    counts = df["keyword"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="#5C85D6")
    ax.set_title("Articles per Search Keyword", fontsize=14, fontweight="bold")
    ax.set_xlabel("Keyword")
    ax.set_ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    _save("03_keyword_distribution.png")


# ── 04: Token length distribution ────────────────────────────────────────────

def plot_token_length_distribution(df: pd.DataFrame) -> None:
    """Histogram: token counts per article with median line."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["token_count"], bins=40, color="#5C85D6", edgecolor="white")
    median = df["token_count"].median()
    ax.axvline(median, color="#F44336", linestyle="--",
               label=f"Median: {median:.0f} tokens")
    ax.set_title("Token Length Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    _save("04_token_length_distribution.png")


# ── 05: Word clouds by sentiment ──────────────────────────────────────────────

def plot_wordcloud(df: pd.DataFrame, label_col: str = "vader_label") -> None:
    """Side-by-side word clouds for Positive, Neutral, Negative articles."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, label in zip(axes, ["Positive", "Neutral", "Negative"]):
        subset = df[df[label_col] == label]["cleaned_text"].dropna()
        text   = " ".join(subset)
        if not text.strip():
            ax.axis("off")
            ax.set_title(label)
            continue
        wc = WordCloud(
            width=600, height=400,
            background_color="white",
            colormap="RdYlGn" if label != "Negative" else "Reds",
            max_words=100,
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{label} Articles", fontsize=13, fontweight="bold",
                     color=_COLORS.get(label, "black"))
    fig.suptitle("Word Clouds by Sentiment Class", fontsize=15, fontweight="bold")
    _save("05_wordcloud_by_sentiment.png")


# ── 06: Word cloud — all articles ─────────────────────────────────────────────

def plot_wordcloud_all(df: pd.DataFrame) -> None:
    """Single word cloud for all cleaned articles."""
    text = " ".join(df["cleaned_text"].dropna())
    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        colormap="Blues",
        max_words=150,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Frequent Words — All Articles", fontsize=15, fontweight="bold")
    _save("06_wordcloud_all.png")


# ── 07: Sentiment distribution ────────────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    """Side-by-side bar charts: VADER vs RoBERTa label counts."""
    has_roberta = "roberta_label" in df.columns
    n_cols = 2 if has_roberta else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    for ax, col, title in zip(
        axes,
        ["vader_label", "roberta_label"][:n_cols],
        ["VADER", "RoBERTa"][:n_cols],
    ):
        counts = df[col].value_counts().reindex(["Positive", "Neutral", "Negative"],
                                                 fill_value=0)
        colors = [_COLORS[l] for l in counts.index]
        counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
        ax.set_title(f"{title} Sentiment Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Article Count")
        ax.tick_params(axis="x", rotation=0)

        for p in ax.patches:
            pct = 100 * p.get_height() / len(df)
            ax.annotate(f"{pct:.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)

    fig.suptitle("Sentiment Distribution Comparison", fontsize=15, fontweight="bold")
    _save("07_sentiment_distribution.png")


# ── 08: Sentiment over time ───────────────────────────────────────────────────

def plot_sentiment_over_time(daily_df: pd.DataFrame) -> None:
    """Line chart: daily VADER compound score with shaded zones."""
    daily = daily_df.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(daily["date"], daily["vader_compound"], color="#5C85D6",
            linewidth=2, label="VADER Compound")
    ax.axhline(0.05,  color="#4CAF50", linestyle="--", alpha=0.6, label="Positive threshold")
    ax.axhline(-0.05, color="#F44336", linestyle="--", alpha=0.6, label="Negative threshold")
    ax.fill_between(daily["date"], daily["vader_compound"], 0.05,
                    where=daily["vader_compound"] >= 0.05,
                    alpha=0.15, color="#4CAF50")
    ax.fill_between(daily["date"], daily["vader_compound"], -0.05,
                    where=daily["vader_compound"] <= -0.05,
                    alpha=0.15, color="#F44336")

    ax.set_title("Daily Sentiment Score Over Time (VADER)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Compound Score")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    ax.legend()
    _save("08_sentiment_over_time.png")


# ── 09: Sentiment vs oil price ────────────────────────────────────────────────

def plot_sentiment_vs_oil(merged_df: pd.DataFrame) -> None:
    """Dual-axis: Brent price + VADER compound score."""
    df = merged_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    ax1.plot(df["date"], df["Brent_Close"], color="#FF7043", linewidth=2, label="Brent (USD)")
    ax2.plot(df["date"], df["vader_compound"], color="#5C85D6", linewidth=1.5,
             linestyle="--", alpha=0.8, label="VADER Compound")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Brent Price (USD/bbl)", color="#FF7043")
    ax2.set_ylabel("VADER Compound Score", color="#5C85D6")
    ax1.tick_params(axis="y", labelcolor="#FF7043")
    ax2.tick_params(axis="y", labelcolor="#5C85D6")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Brent Crude Price vs. VADER Sentiment", fontsize=14, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    _save("09_sentiment_vs_oil_price.png")


# ── 10: Correlation scatter ───────────────────────────────────────────────────

def plot_correlation_scatter(merged_df: pd.DataFrame) -> None:
    """Scatter plots: sentiment vs Brent and WTI with regression lines."""
    df = merged_df.dropna()
    models = [("vader_compound", "VADER")]
    if "roberta_compound" in df.columns:
        models.append(("roberta_compound", "RoBERTa"))

    cols = [("Brent_Close", "Brent"), ("WTI_Close", "WTI")]
    n   = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n))
    if n == 1:
        axes = [axes]

    for row, (sent_col, sent_name) in enumerate(models):
        for col_idx, (price_col, price_name) in enumerate(cols):
            ax = axes[row][col_idx]
            x  = df[sent_col]
            y  = df[price_col]
            r  = x.corr(y)

            ax.scatter(x, y, alpha=0.6, color="#5C85D6", edgecolors="white", s=50)
            m, b = np.polyfit(x, y, 1)
            xline = np.linspace(x.min(), x.max(), 100)
            ax.plot(xline, m * xline + b, color="#F44336", linewidth=2)

            ax.set_title(f"{sent_name} vs {price_name}  (r = {r:.4f})",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel(f"{sent_name} Compound")
            ax.set_ylabel(f"{price_name} Price (USD)")

    fig.suptitle("Sentiment vs Crude Oil Price — Scatter Plots", fontsize=15, fontweight="bold")
    _save("10_correlation_scatter.png")


# ── 11: Rolling correlation ───────────────────────────────────────────────────

def plot_rolling_correlation(merged_df: pd.DataFrame, window: int = 7) -> None:
    """7-day rolling Pearson correlation: VADER vs Brent."""
    df = merged_df.copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    fig, ax = plt.subplots(figsize=(13, 4))

    pairs = [("vader_compound", "Brent_Close", "#5C85D6", "VADER / Brent")]
    if "roberta_compound" in df.columns:
        pairs.append(("roberta_compound", "Brent_Close", "#FF7043", "RoBERTa / Brent"))

    for sent_col, price_col, color, label in pairs:
        if sent_col in df.columns and price_col in df.columns:
            roll_r = df[sent_col].rolling(window).corr(df[price_col])
            ax.plot(roll_r.index, roll_r, linewidth=2, color=color, label=label)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{window}-Day Rolling Correlation: Sentiment vs Brent Crude",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Pearson r")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    ax.legend()
    _save("11_rolling_correlation.png")


# ── 12: Model agreement ───────────────────────────────────────────────────────

def plot_model_agreement(df: pd.DataFrame) -> None:
    """Agreement rate and confusion-style heatmap between VADER and RoBERTa."""
    if "roberta_label" not in df.columns:
        print("  [skip] roberta_label not found — skipping model agreement plot")
        return

    agree = (df["vader_label"] == df["roberta_label"]).mean() * 100

    confusion = pd.crosstab(
        df["vader_label"], df["roberta_label"],
        rownames=["VADER"], colnames=["RoBERTa"],
    ).reindex(index=["Positive", "Neutral", "Negative"],
              columns=["Positive", "Neutral", "Negative"], fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Article Count"})
    ax.set_title(f"Model Agreement: {agree:.1f}%  |  VADER vs RoBERTa",
                 fontsize=13, fontweight="bold")
    _save("12_model_agreement.png")
