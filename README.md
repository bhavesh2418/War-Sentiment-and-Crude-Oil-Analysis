# War Sentiment & Crude Oil Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue)
![NLP](https://img.shields.io/badge/NLP-VADER%20%2B%20RoBERTa-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

**Does war-related media sentiment predict crude oil price movements? VADER says yes (r = +0.65). RoBERTa says no — and that gap is the most important finding.**

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Workflow](#workflow)
5. [EDA](#eda)
6. [Sentiment Analysis](#sentiment-analysis)
7. [Results](#results)
8. [Key Insights](#key-insights)
9. [Setup & Usage](#setup--usage)
10. [Tech Stack](#tech-stack)

---

## Problem Statement

This project investigates whether media sentiment about the US–Israel–Iran conflict correlates with Brent and WTI crude oil prices. Data comes from Google News RSS and NewsAPI (no paid sources), processed through two complementary NLP pipelines — VADER and Twitter-RoBERTa — then correlated against commodity futures via yfinance.

---

## Dataset

| Feature | Type | Description |
|---|---|---|
| title | text | Article headline |
| text | text | Article body / description |
| source | categorical | Publisher name |
| keyword | categorical | Search term that retrieved the article |
| published | datetime | UTC publication timestamp |
| cleaned_text | text | Preprocessed text (tokens, lemmas) |
| token_count | int | Number of tokens after cleaning |
| vader_compound | float | VADER compound sentiment score (-1 to +1) |
| vader_label | categorical | Positive / Neutral / Negative |
| roberta_compound | float | RoBERTa pos-neg probability difference |
| roberta_label | categorical | Positive / Neutral / Negative |
| Brent_Close | float | Brent crude daily closing price (USD) |
| WTI_Close | float | WTI crude daily closing price (USD) |

**Sources:** Google News RSS + NewsAPI | **Period:** Nov 2025 – Apr 2026 | **Articles:** ~796 (after deduplication)

---

## Project Structure

```
War-Sentiment-and-Crude-Oil-Analysis/
├── data/
│   ├── raw/               # news_raw.csv, crude_oil_prices.csv
│   └── processed/         # news_processed.csv
├── notebooks/             # 6 Jupyter notebooks (one per phase)
├── src/                   # Python source modules
│   ├── config.py          # All paths, constants, model params
│   ├── data_collector.py  # Google News RSS + NewsAPI + yfinance
│   ├── preprocessing.py   # Text cleaning pipeline
│   ├── sentiment.py       # VADER + RoBERTa + daily aggregation
│   └── visualize.py       # All 13 plot functions
├── models/                # RoBERTa tokenizer/config (downloaded once)
├── images/                # All plots — committed for README display
├── reports/               # Results CSVs + PDF report
├── scripts/
│   ├── collect_data.py    # Standalone data refresh
│   └── generate_pdf.py    # PDF report generator
├── main.py                # Full pipeline runner
└── requirements.txt
```

---

## Workflow

```
Google News RSS  +  NewsAPI  +  yfinance
        |                |           |
   feedparser       requests    yf.download
         \              /            |
          news_raw.csv     crude_oil_prices.csv
               |
     Text Cleaning (8 steps)
               |
     news_processed.csv
          /         \
    VADER NLP     RoBERTa NLP
  (lexicon)     (transformer)
         \           /
    Daily Aggregation
              |
    Merge with Oil Prices
              |
  Pearson r + Rolling Correlation
              |
    13 Visualizations + PDF Report
```

---

## EDA

### Crude Oil Trend
![Crude Oil Trend](images/00_crude_oil_trend.png)

### Articles Over Time
![Articles Over Time](images/01_articles_over_time.png)

### Source Distribution
![Source Distribution](images/02_source_distribution.png)

### Word Cloud — All Articles
![Word Cloud All](images/06_wordcloud_all.png)

---

## Sentiment Analysis

### Sentiment Distribution (VADER vs RoBERTa)
![Sentiment Distribution](images/07_sentiment_distribution.png)

The key finding: **VADER classifies ~68% of articles as Negative**. RoBERTa classifies **~63% as Neutral**. Most war journalism maintains factual, neutral tone — context-aware RoBERTa sees this, but misses the vocabulary signals traders respond to.

### Sentiment Over Time
![Sentiment Over Time](images/08_sentiment_over_time.png)

### Word Clouds by Sentiment
![Word Cloud by Sentiment](images/05_wordcloud_by_sentiment.png)

---

## Results

### Correlation — Sentiment vs Crude Oil Prices

| Model | Market | Pearson r | Interpretation |
|---|---|---|---|
| VADER | Brent | **+0.6473** | Strong positive correlation |
| VADER | WTI | **+0.5891** | Moderate positive correlation |
| RoBERTa | Brent | +0.0642 | Weak positive correlation |
| RoBERTa | WTI | +0.0521 | Weak positive correlation |

### Sentiment vs Brent Price
![Sentiment vs Oil](images/09_sentiment_vs_oil_price.png)

### Correlation Scatter Plots
![Correlation Scatter](images/10_correlation_scatter.png)

### 7-Day Rolling Correlation
![Rolling Correlation](images/11_rolling_correlation.png)

### Model Agreement
![Model Agreement](images/12_model_agreement.png)

---

## Key Insights

1. **VADER r = +0.647 with Brent crude** — vocabulary-level negativity intensity tracks oil price movements
2. **RoBERTa r = +0.064** — transformer models underweight raw headline language markets respond to
3. **"VADER reacts to vocabulary. RoBERTa understands context."** — For commodity correlation, vocabulary wins
4. Article volume spikes precede Brent price volatility by 1-2 days
5. Rolling correlation is strongest during escalation events, weakest during diplomatic pauses
6. Free data sources (RSS + NewsAPI + yfinance) are sufficient for meaningful geopolitical NLP research

---

## Setup & Usage

```bash
# 1. Clone
git clone https://github.com/bhavesh2418/War-Sentiment-and-Crude-Oil-Analysis.git
cd War-Sentiment-and-Crude-Oil-Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set credentials — create a .env file:
#    NEWS_API_KEY=your_key_here

# 4. Collect data
python scripts/collect_data.py

# 5. Run full pipeline
python main.py

# 6. VADER-only (faster, no GPU needed)
python main.py --skip-roberta

# 7. Use cached data
python main.py --use-cache

# 8. Generate PDF report
python scripts/generate_pdf.py
```

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| pandas | 2.2.3 | Data manipulation |
| numpy | 2.2.6 | Numerical operations |
| feedparser | 6.0.12 | Google News RSS parsing |
| requests | 2.32.3 | NewsAPI REST calls |
| yfinance | 1.2.1 | Crude oil price download |
| nltk | 3.9.4 | Tokenization, stopwords, lemmatization |
| vaderSentiment | 3.3.2 | Rule-based sentiment scoring |
| transformers | 5.5.0 | CardiffNLP Twitter-RoBERTa |
| torch | 2.11.0 | Transformer inference backend |
| matplotlib | 3.10.0 | Visualizations |
| seaborn | 0.13.2 | Statistical plots |
| wordcloud | 1.9.6 | Word cloud generation |
| fpdf2 | 2.8.7 | PDF report generation |
| python-dotenv | 1.1.0 | Environment variable loading |
