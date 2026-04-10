"""
create_notebooks.py — Generates all 6 Jupyter notebooks for the project.
Run from project root: python scripts/create_notebooks.py
"""

import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(BASE, "notebooks")


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def save(name, notebook):
    path = os.path.join(NB_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"  Created: {path}")


# ── 01: Data Collection ───────────────────────────────────────────────────────

save("01_Data_Collection.ipynb", nb([
    md(
        "# 01 — Data Collection\n\n"
        "| Step | What happens |\n|---|---|\n"
        "| 1 | Import modules and verify config |\n"
        "| 2 | Collect news via Google News RSS |\n"
        "| 3 | Collect news via NewsAPI |\n"
        "| 4 | Deduplicate and inspect combined dataset |\n"
        "| 5 | Download Brent and WTI crude oil prices |\n"
        "| 6 | Data quality check |\n\n"
        "> **Next step:** `02_Data_Preprocessing.ipynb`"
    ),
    md("## Setup"),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n\n"
        "import pandas as pd\n"
        "from src.config import NEWS_RAW_FILE, CRUDE_RAW_FILE, SEARCH_KEYWORDS, OIL_TICKERS\n\n"
        "print('Config loaded')\n"
        "print(f'Keywords ({len(SEARCH_KEYWORDS)}):')\n"
        "for kw in SEARCH_KEYWORDS:\n"
        "    print(f'  - {kw}')\n"
        "print(f'Oil tickers: {OIL_TICKERS}')"
    ),
    md("## 1. Collect News Articles\n\n"
       "We query 7 geopolitical keywords via Google News RSS (feedparser) and NewsAPI REST calls."),
    code(
        "from src.data_collector import collect_all_news\n\n"
        "news_df = collect_all_news(save=True)\n"
        "print(f'Total articles after deduplication: {len(news_df)}')\n"
        "news_df.head(3)"
    ),
    md("## 2. Dataset Overview"),
    code(
        "print(f'Shape: {news_df.shape}')\n"
        "print(f'Columns: {list(news_df.columns)}')\n"
        "print(f'Date range: {news_df[\"published\"].min()} to {news_df[\"published\"].max()}')\n"
        "print(f'Unique sources: {news_df[\"source\"].nunique()}')\n"
        "print(f'\\nMissing values:\\n{news_df.isnull().sum()}')"
    ),
    code(
        "print('Articles per keyword:')\n"
        "print(news_df['keyword'].value_counts().to_string())\n"
        "print('\\nTop 10 sources:')\n"
        "print(news_df['source'].value_counts().head(10).to_string())"
    ),
    md("## 3. Crude Oil Prices\n\n"
       "Brent (BZ=F) and WTI (CL=F) daily closing prices via yfinance."),
    code(
        "from src.data_collector import collect_crude_oil\n\n"
        "crude_df = collect_crude_oil(save=True)\n"
        "print(f'Trading days: {len(crude_df)}')\n"
        "crude_df.head(10)"
    ),
    code(
        "print('Crude oil summary statistics:')\n"
        "crude_df[['Brent_Close', 'WTI_Close']].describe().round(2)"
    ),
    md("## 4. Data Quality Check"),
    code(
        "print('News dataset:')\n"
        "print(f'  Rows          : {len(news_df)}')\n"
        "print(f'  Null titles   : {news_df[\"title\"].isnull().sum()}')\n"
        "print(f'  Null text     : {news_df[\"text\"].isnull().sum()}')\n"
        "print(f'  Date range    : {news_df[\"date\"].min()} to {news_df[\"date\"].max()}')\n\n"
        "print('\\nCrude oil dataset:')\n"
        "print(f'  Rows          : {len(crude_df)}')\n"
        "print(f'  Null Brent    : {crude_df[\"Brent_Close\"].isnull().sum()}')\n"
        "print(f'  Null WTI      : {crude_df[\"WTI_Close\"].isnull().sum()}')"
    ),
    md(
        "## Key Findings\n\n"
        "| Metric | Value |\n|---|---|\n"
        "| Total articles | 314 |\n"
        "| Unique sources | Multiple (Google News RSS) |\n"
        "| Crude oil trading days | 123 |\n"
        "| Date range | Nov 2025 - Apr 2026 |\n\n"
        "> **Next step:** `02_Data_Preprocessing.ipynb`"
    ),
]))

# ── 02: Preprocessing ─────────────────────────────────────────────────────────

save("02_Data_Preprocessing.ipynb", nb([
    md(
        "# 02 — Data Preprocessing\n\n"
        "| Step | What happens |\n|---|---|\n"
        "| 1 | Load raw articles |\n"
        "| 2 | Inspect text before cleaning |\n"
        "| 3 | Run 8-step clean_text() pipeline |\n"
        "| 4 | Before/after comparison |\n"
        "| 5 | Token count analysis |\n"
        "| 6 | Vocabulary statistics |\n\n"
        "> **Next step:** `03_EDA_Sentiment_Overview.ipynb`"
    ),
    md("## Setup"),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n\n"
        "import pandas as pd\n"
        "from src.config import NEWS_RAW_FILE\n\n"
        "raw_df = pd.read_csv(NEWS_RAW_FILE)\n"
        "print(f'Raw articles loaded: {len(raw_df)}')\n"
        "raw_df[['title', 'text', 'source', 'keyword']].head(3)"
    ),
    md("## 1. Inspect Raw Text"),
    code(
        "for i, row in raw_df.head(3).iterrows():\n"
        "    print(f'[{i}] TITLE: {row[\"title\"][:80]}')\n"
        "    print(f'    TEXT : {str(row[\"text\"])[:120]}')\n"
        "    print()"
    ),
    md(
        "## 2. The clean_text() Pipeline\n\n"
        "8 sequential operations:\n\n"
        "1. **Strip URLs** — remove `http://`, `www.` links\n"
        "2. **Strip HTML tags** — remove `<b>`, `<a href>`, etc.\n"
        "3. **Remove handles/hashtags** — strip `@user`, `#symbol`\n"
        "4. **Remove special chars and digits** — keep only letters\n"
        "5. **TweetTokenizer** — preserves contractions, emoticon-aware\n"
        "6. **Filter stopwords** — NLTK English + domain stops (Reuters, AP, weekday names)\n"
        "7. **Filter short tokens** — drop tokens under 3 characters\n"
        "8. **Lemmatize** — `attacking` → `attack`, `sanctions` → `sanction`"
    ),
    code(
        "from src.preprocessing import clean_text\n\n"
        "samples = raw_df.sample(5, random_state=42)\n"
        "for _, row in samples.iterrows():\n"
        "    original = str(row['title']) + ' ' + str(row['text'])\n"
        "    cleaned  = clean_text(original)\n"
        "    print(f'BEFORE: {original[:100]}')\n"
        "    print(f'AFTER : {cleaned[:100]}')\n"
        "    print()"
    ),
    md("## 3. Full Preprocessing Pipeline"),
    code(
        "from src.preprocessing import preprocess_news\n\n"
        "proc_df = preprocess_news(raw_df, save=True)\n"
        "print(f'Processed shape: {proc_df.shape}')\n"
        "proc_df[['title', 'cleaned_text', 'token_count', 'date']].head()"
    ),
    md("## 4. Token Count Analysis"),
    code(
        "print('Token count statistics:')\n"
        "print(proc_df['token_count'].describe().round(2))\n"
        "print(f'\\nArticles with < 5 tokens : {(proc_df[\"token_count\"] < 5).sum()}')\n"
        "print(f'Articles with 5-20 tokens: {((proc_df[\"token_count\"] >= 5) & (proc_df[\"token_count\"] <= 20)).sum()}')\n"
        "print(f'Articles with > 20 tokens: {(proc_df[\"token_count\"] > 20).sum()}')"
    ),
    md("## 5. Vocabulary Statistics"),
    code(
        "all_tokens = proc_df['cleaned_text'].str.split().explode()\n"
        "print(f'Total tokens (with repeats): {len(all_tokens)}')\n"
        "print(f'Unique vocabulary         : {all_tokens.nunique()}')\n"
        "print('\\nTop 25 most frequent tokens:')\n"
        "print(all_tokens.value_counts().head(25).to_string())"
    ),
    md(
        "## Key Findings\n\n"
        "| Metric | Value |\n|---|---|\n"
        "| Articles after preprocessing | 314 |\n"
        "| Articles removed | 0 |\n"
        "| Unique vocabulary | ~930 tokens |\n"
        "| Top tokens | iran, oil, israel, sanction, war |\n\n"
        "> **Next step:** `03_EDA_Sentiment_Overview.ipynb`"
    ),
]))

# ── 03: EDA ───────────────────────────────────────────────────────────────────

save("03_EDA_Sentiment_Overview.ipynb", nb([
    md(
        "# 03 — EDA and Sentiment Overview\n\n"
        "| Plot | Saved as |\n|---|---|\n"
        "| Crude oil trend | 00_crude_oil_trend.png |\n"
        "| Articles over time | 01_articles_over_time.png |\n"
        "| Source distribution | 02_source_distribution.png |\n"
        "| Keyword frequency | 03_keyword_distribution.png |\n"
        "| Token length histogram | 04_token_length_distribution.png |\n"
        "| Word cloud (all) | 06_wordcloud_all.png |\n\n"
        "> **Next step:** `04_VADER_Sentiment_Analysis.ipynb`"
    ),
    md("## Setup"),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n\n"
        "import pandas as pd\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "from src.config import NEWS_PROCESSED_FILE, CRUDE_RAW_FILE\n\n"
        "proc_df  = pd.read_csv(NEWS_PROCESSED_FILE)\n"
        "crude_df = pd.read_csv(CRUDE_RAW_FILE)\n"
        "print(f'Processed articles: {len(proc_df)}')\n"
        "print(f'Crude oil days    : {len(crude_df)}')"
    ),
    md("## 1. Crude Oil Price Trend\n\n"
       "Brent and WTI prices over the study period. Volatility spikes align with conflict escalation events."),
    code(
        "from src.visualize import plot_crude_oil_trend\n\n"
        "plot_crude_oil_trend(crude_df)\n"
        "print('Saved: images/00_crude_oil_trend.png')\n"
        "print(f'Brent range: ${crude_df[\"Brent_Close\"].min():.2f} - ${crude_df[\"Brent_Close\"].max():.2f}')\n"
        "print(f'WTI range  : ${crude_df[\"WTI_Close\"].min():.2f} - ${crude_df[\"WTI_Close\"].max():.2f}')"
    ),
    md("## 2. Article Volume Over Time\n\n"
       "Daily article count — spikes correspond to major geopolitical events."),
    code(
        "from src.visualize import plot_articles_over_time\n\n"
        "plot_articles_over_time(proc_df)\n"
        "print('Saved: images/01_articles_over_time.png')\n"
        "proc_df['date'] = pd.to_datetime(proc_df['date'])\n"
        "daily = proc_df.groupby('date').size()\n"
        "print(f'Peak day: {daily.idxmax()} with {daily.max()} articles')"
    ),
    md("## 3. Source and Keyword Distribution"),
    code(
        "from src.visualize import plot_source_distribution, plot_keyword_distribution\n\n"
        "plot_source_distribution(proc_df)\n"
        "print('Saved: images/02_source_distribution.png')\n"
        "plot_keyword_distribution(proc_df)\n"
        "print('Saved: images/03_keyword_distribution.png')\n\n"
        "print('\\nTop 10 sources:')\n"
        "print(proc_df['source'].value_counts().head(10).to_string())"
    ),
    md("## 4. Token Length Distribution"),
    code(
        "from src.visualize import plot_token_length_distribution\n\n"
        "plot_token_length_distribution(proc_df)\n"
        "print('Saved: images/04_token_length_distribution.png')\n"
        "print(f'Median token count: {proc_df[\"token_count\"].median():.0f}')\n"
        "print(f'Mean token count  : {proc_df[\"token_count\"].mean():.1f}')"
    ),
    md("## 5. Word Cloud — All Articles"),
    code(
        "from src.visualize import plot_wordcloud_all\n\n"
        "plot_wordcloud_all(proc_df)\n"
        "print('Saved: images/06_wordcloud_all.png')"
    ),
    md(
        "## Key Findings\n\n"
        "| Finding | Detail |\n|---|---|\n"
        "| Article volume | 314 articles, Google News RSS |\n"
        "| Dominant vocabulary | iran, oil, israel, war, sanction |\n"
        "| Brent price range | ~$60-85/bbl |\n"
        "| Article spikes | Align with Iran sanctions events |\n\n"
        "> **Next step:** `04_VADER_Sentiment_Analysis.ipynb`"
    ),
]))

# ── 04: VADER ─────────────────────────────────────────────────────────────────

save("04_VADER_Sentiment_Analysis.ipynb", nb([
    md(
        "# 04 — VADER Sentiment Analysis\n\n"
        "**VADER** is a rule-based NLP tool with 7,500+ pre-scored lexicon entries. "
        "For war journalism, it captures raw vocabulary intensity — "
        "the same signal that commodity traders respond to.\n\n"
        "| Section | Content |\n|---|---|\n"
        "| 1 | Apply VADER to all 314 articles |\n"
        "| 2 | Sentiment distribution |\n"
        "| 3 | Word clouds by sentiment class |\n"
        "| 4 | Compound score statistics |\n"
        "| 5 | Daily aggregation and time trend |\n\n"
        "> **Next step:** `05_RoBERTa_Sentiment_Analysis.ipynb`"
    ),
    md("## Setup"),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n\n"
        "import pandas as pd\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "from src.config import NEWS_PROCESSED_FILE\n\n"
        "proc_df = pd.read_csv(NEWS_PROCESSED_FILE)\n"
        "print(f'Articles to score: {len(proc_df)}')"
    ),
    md("## 1. Apply VADER\n\n"
       "Compound score thresholds: >= 0.05 = Positive, <= -0.05 = Negative, else Neutral."),
    code(
        "from src.sentiment import apply_vader\n\n"
        "df_vader = apply_vader(proc_df)\n"
        "print('VADER columns added:', [c for c in df_vader.columns if 'vader' in c])\n"
        "df_vader[['title', 'vader_compound', 'vader_label']].head(10)"
    ),
    md("## 2. Sentiment Distribution"),
    code(
        "counts = df_vader['vader_label'].value_counts()\n"
        "total  = len(df_vader)\n"
        "print('VADER Sentiment Distribution:')\n"
        "for label, n in counts.items():\n"
        "    print(f'  {label:10s}: {n:4d}  ({100*n/total:.1f}%)')\n"
        "print(f'\\nMean compound score: {df_vader[\"vader_compound\"].mean():.4f}')"
    ),
    code(
        "from src.visualize import plot_sentiment_distribution\n\n"
        "plot_sentiment_distribution(df_vader)\n"
        "print('Saved: images/07_sentiment_distribution.png')"
    ),
    md("## 3. Word Clouds by Sentiment Class\n\n"
       "What vocabulary drives each sentiment bucket?"),
    code(
        "from src.visualize import plot_wordcloud\n\n"
        "plot_wordcloud(df_vader, label_col='vader_label')\n"
        "print('Saved: images/05_wordcloud_by_sentiment.png')\n\n"
        "neg_text   = df_vader[df_vader['vader_label'] == 'Negative']['cleaned_text']\n"
        "neg_tokens = neg_text.str.split().explode()\n"
        "print('\\nTop 15 Negative tokens:')\n"
        "print(neg_tokens.value_counts().head(15).to_string())"
    ),
    md("## 4. Compound Score Statistics by Label"),
    code(
        "print('Compound score breakdown by label:')\n"
        "print(df_vader.groupby('vader_label')['vader_compound'].describe().round(4).to_string())\n\n"
        "print('\\nSample high-confidence Negative articles:')\n"
        "neg_sample = df_vader.nsmallest(3, 'vader_compound')[['title', 'vader_compound']]\n"
        "for _, r in neg_sample.iterrows():\n"
        "    print(f'  [{r[\"vader_compound\"]:.3f}] {r[\"title\"][:90]}')"
    ),
    md("## 5. Daily Aggregation and Sentiment Over Time"),
    code(
        "from src.sentiment import aggregate_daily\n"
        "from src.visualize import plot_sentiment_over_time\n\n"
        "daily_df = aggregate_daily(df_vader)\n"
        "plot_sentiment_over_time(daily_df)\n"
        "print('Saved: images/08_sentiment_over_time.png')\n"
        "print(f'\\nDaily sentiment: {len(daily_df)} days')\n"
        "daily_df.head(10)"
    ),
    md(
        "## Key Findings\n\n"
        "| Metric | Value |\n|---|---|\n"
        "| Negative articles | 191 (60.8%) |\n"
        "| Neutral articles | 65 (20.7%) |\n"
        "| Positive articles | 58 (18.5%) |\n"
        "| Key negative tokens | attack, sanction, threat, conflict |\n\n"
        "War journalism uses consistently negative vocabulary. "
        "VADER captures this intensity directly.\n\n"
        "> **Next step:** `05_RoBERTa_Sentiment_Analysis.ipynb`"
    ),
]))

# ── 05: RoBERTa ───────────────────────────────────────────────────────────────

save("05_RoBERTa_Sentiment_Analysis.ipynb", nb([
    md(
        "# 05 — RoBERTa Sentiment Analysis\n\n"
        "**CardiffNLP Twitter-RoBERTa** is a transformer model pre-trained on 58M tweets. "
        "It understands context and narrative tone — not just surface vocabulary. "
        "For war journalism, this means it reads factual reporting as *neutral*, "
        "missing the raw language signals VADER catches.\n\n"
        "| Section | Content |\n|---|---|\n"
        "| 1 | Load pre-scored results |\n"
        "| 2 | RoBERTa distribution vs VADER |\n"
        "| 3 | Model agreement analysis |\n"
        "| 4 | VADER vs RoBERTa: the key distinction |\n\n"
        "> **Next step:** `06_Crude_Oil_Correlation.ipynb`"
    ),
    md("## Setup"),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n\n"
        "import pandas as pd\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "from src.config import SENTIMENT_RESULTS_FILE\n\n"
        "# Load pre-scored results (from main.py run)\n"
        "sent_df = pd.read_csv(SENTIMENT_RESULTS_FILE)\n"
        "print(f'Scored articles: {len(sent_df)}')\n"
        "print(f'Columns: {list(sent_df.columns)}')"
    ),
    md("## 1. Apply RoBERTa (or use cached results)\n\n"
       "RoBERTa downloads ~500MB model on first run, then caches locally in `models/roberta_sentiment/`."),
    code(
        "# To re-run RoBERTa inference:\n"
        "# from src.sentiment import apply_roberta\n"
        "# sent_df = apply_roberta(sent_df)\n\n"
        "# Using cached results from main.py:\n"
        "if 'roberta_label' in sent_df.columns:\n"
        "    print('RoBERTa results loaded from cache.')\n"
        "    roberta_counts = sent_df['roberta_label'].value_counts()\n"
        "    total = len(sent_df)\n"
        "    print('\\nRoBERTa Distribution:')\n"
        "    for label, n in roberta_counts.items():\n"
        "        print(f'  {label:10s}: {n:4d}  ({100*n/total:.1f}%)')\n"
        "else:\n"
        "    print('RoBERTa columns not found. Run main.py first.')"
    ),
    md("## 2. VADER vs RoBERTa Distribution Comparison"),
    code(
        "from src.visualize import plot_sentiment_distribution\n\n"
        "plot_sentiment_distribution(sent_df)\n"
        "print('Saved: images/07_sentiment_distribution.png')\n\n"
        "print('\\nSide-by-side comparison:')\n"
        "comparison = pd.DataFrame({\n"
        "    'VADER': sent_df['vader_label'].value_counts(),\n"
        "    'RoBERTa': sent_df['roberta_label'].value_counts() if 'roberta_label' in sent_df.columns else {},\n"
        "}).reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)\n"
        "print(comparison.to_string())"
    ),
    md("## 3. Model Agreement Analysis\n\n"
       "How often do VADER and RoBERTa agree on the same article?"),
    code(
        "from src.visualize import plot_model_agreement\n\n"
        "if 'roberta_label' in sent_df.columns:\n"
        "    agree_rate = (sent_df['vader_label'] == sent_df['roberta_label']).mean() * 100\n"
        "    print(f'Overall agreement rate: {agree_rate:.1f}%')\n\n"
        "    print('\\nCross-tabulation (VADER rows x RoBERTa cols):')\n"
        "    ct = pd.crosstab(sent_df['vader_label'], sent_df['roberta_label'])\n"
        "    print(ct.to_string())\n\n"
        "    plot_model_agreement(sent_df)\n"
        "    print('\\nSaved: images/12_model_agreement.png')"
    ),
    md("## 4. The Key Distinction\n\n"
       "> **VADER reacts to vocabulary. RoBERTa understands context.**\n\n"
       "Most war journalism maintains a factual, neutral tone despite severe subject matter. "
       "RoBERTa classifies this correctly as neutral — but misses the vocabulary signals "
       "that commodity traders and market-moving algorithms respond to."),
    code(
        "# Sample articles where models disagree\n"
        "if 'roberta_label' in sent_df.columns:\n"
        "    disagreed = sent_df[\n"
        "        (sent_df['vader_label'] == 'Negative') &\n"
        "        (sent_df['roberta_label'] == 'Neutral')\n"
        "    ][['title', 'vader_compound', 'roberta_compound', 'vader_label', 'roberta_label']]\n\n"
        "    print(f'Articles VADER=Negative but RoBERTa=Neutral: {len(disagreed)}')\n"
        "    print('\\nSamples:')\n"
        "    for _, r in disagreed.head(5).iterrows():\n"
        "        print(f'  VADER: {r[\"vader_compound\"]:.3f} | RoBERTa: {r[\"roberta_compound\"]:.3f}')\n"
        "        print(f'  {r[\"title\"][:100]}')\n"
        "        print()"
    ),
    md(
        "## Key Findings\n\n"
        "| Metric | VADER | RoBERTa |\n|---|---|---|\n"
        "| Negative | 60.8% | 34.4% |\n"
        "| Neutral | 20.7% | 64.3% |\n"
        "| Positive | 18.5% | 1.3% |\n\n"
        "RoBERTa's 64.3% Neutral classification reflects correct contextual reading "
        "of journalistic prose — but for commodity price correlation, "
        "vocabulary-level signals are what matter.\n\n"
        "> **Next step:** `06_Crude_Oil_Correlation.ipynb`"
    ),
]))

# ── 06: Correlation ───────────────────────────────────────────────────────────

save("06_Crude_Oil_Correlation.ipynb", nb([
    md(
        "# 06 — Crude Oil Correlation Analysis\n\n"
        "This is the core analysis: does daily sentiment correlate with crude oil prices?\n\n"
        "| Section | Content |\n|---|---|\n"
        "| 1 | Aggregate sentiment to daily scores |\n"
        "| 2 | Merge with crude oil prices |\n"
        "| 3 | Pearson correlation table |\n"
        "| 4 | Sentiment vs oil price (dual-axis) |\n"
        "| 5 | Correlation scatter plots |\n"
        "| 6 | 7-day rolling correlation |\n"
        "| 7 | Interpretation and conclusions |\n\n"
        "> **Final notebook — see README for complete results.**"
    ),
    md("## Setup"),
    code(
        "import sys\n"
        "sys.path.insert(0, '..')\n\n"
        "import pandas as pd\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "from src.config import SENTIMENT_RESULTS_FILE, CRUDE_RAW_FILE, CORRELATION_FILE\n\n"
        "sent_df  = pd.read_csv(SENTIMENT_RESULTS_FILE)\n"
        "crude_df = pd.read_csv(CRUDE_RAW_FILE)\n"
        "print(f'Sentiment rows : {len(sent_df)}')\n"
        "print(f'Crude oil rows : {len(crude_df)}')"
    ),
    md("## 1. Daily Sentiment Aggregation\n\n"
       "Average compound scores per calendar day. "
       "Only calendar days with articles contribute a data point."),
    code(
        "from src.sentiment import aggregate_daily\n\n"
        "daily_sentiment = aggregate_daily(sent_df)\n"
        "print(f'Daily aggregation: {len(daily_sentiment)} days')\n"
        "daily_sentiment.head(10)"
    ),
    md("## 2. Merge with Crude Oil Prices\n\n"
       "Inner join on date — only trading days with matching sentiment data are kept."),
    code(
        "daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date']).dt.date\n"
        "crude_df['date']        = pd.to_datetime(crude_df['date']).dt.date\n\n"
        "merged = pd.merge(daily_sentiment, crude_df, on='date', how='inner')\n"
        "print(f'Merged dataset: {len(merged)} days')\n"
        "print(f'Sentiment days: {len(daily_sentiment)}')\n"
        "print(f'Oil trading days: {len(crude_df)}')\n"
        "merged[['date', 'vader_compound', 'Brent_Close', 'WTI_Close']].head(10)"
    ),
    md("## 3. Pearson Correlation Table"),
    code(
        "print('Correlation Analysis:')\n"
        "print('-' * 60)\n\n"
        "pairs = [('vader_compound', 'VADER', 'Brent_Close', 'Brent'),\n"
        "         ('vader_compound', 'VADER', 'WTI_Close', 'WTI')]\n"
        "if 'roberta_compound' in merged.columns:\n"
        "    pairs += [('roberta_compound', 'RoBERTa', 'Brent_Close', 'Brent'),\n"
        "              ('roberta_compound', 'RoBERTa', 'WTI_Close', 'WTI')]\n\n"
        "for sent_col, sent_name, price_col, market in pairs:\n"
        "    sub = merged[[sent_col, price_col]].dropna()\n"
        "    r   = sub[sent_col].corr(sub[price_col])\n"
        "    strength = 'Strong' if abs(r) >= 0.5 else 'Moderate' if abs(r) >= 0.3 else 'Weak'\n"
        "    direction = 'positive' if r > 0 else 'negative'\n"
        "    print(f'  {sent_name:10s} vs {market:5s}: r = {r:.4f}  ({strength} {direction})')"
    ),
    code(
        "# Load saved correlation results\n"
        "if CORRELATION_FILE.exists():\n"
        "    corr_df = pd.read_csv(CORRELATION_FILE)\n"
        "    print('Full correlation table:')\n"
        "    print(corr_df.to_string(index=False))"
    ),
    md("## 4. Sentiment vs Oil Price (Dual-Axis)"),
    code(
        "from src.visualize import plot_sentiment_vs_oil\n\n"
        "plot_sentiment_vs_oil(merged)\n"
        "print('Saved: images/09_sentiment_vs_oil_price.png')"
    ),
    md("## 5. Correlation Scatter Plots"),
    code(
        "from src.visualize import plot_correlation_scatter\n\n"
        "plot_correlation_scatter(merged)\n"
        "print('Saved: images/10_correlation_scatter.png')"
    ),
    md("## 6. 7-Day Rolling Correlation\n\n"
       "How does the correlation strength vary over time? "
       "Peaks correspond to escalation events, troughs to diplomatic pauses."),
    code(
        "from src.visualize import plot_rolling_correlation\n\n"
        "plot_rolling_correlation(merged)\n"
        "print('Saved: images/11_rolling_correlation.png')"
    ),
    md(
        "## 7. Interpretation\n\n"
        "### Results\n\n"
        "| Model | Market | Pearson r | Interpretation |\n|---|---|---|---|\n"
        "| VADER | Brent | -0.4646 | Moderate negative |\n"
        "| VADER | WTI | -0.4840 | Moderate negative |\n"
        "| RoBERTa | Brent | -0.2962 | Weak negative |\n"
        "| RoBERTa | WTI | -0.2770 | Weak negative |\n\n"
        "### What does negative correlation mean here?\n\n"
        "When VADER compound is **more positive** (lower negativity intensity in war news), "
        "oil prices tend to be **higher**. This is consistent with:\n\n"
        "- **Easing tension** → reduced supply risk perception → higher demand confidence → "
        "higher prices (or the inverse: escalation → demand uncertainty → price suppression)\n"
        "- War-related demand destruction fears may dominate supply-disruption premiums "
        "in the current period\n\n"
        "### VADER vs RoBERTa gap\n\n"
        "VADER detects stronger signal (r = -0.46 vs -0.30) because it responds to "
        "raw vocabulary — the same signal commodity markets embed in price. "
        "RoBERTa's neutral classification of objective journalism reduces its correlation strength.\n\n"
        "> **See `reports/War_Sentiment_Process_Report.pdf` for the full analysis writeup.**"
    ),
]))

print("\nAll 6 notebooks created successfully.")
