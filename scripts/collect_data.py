"""
collect_data.py — Standalone script to refresh news and crude oil data.
Run from project root: python scripts/collect_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collector import collect_all_news, collect_crude_oil

if __name__ == "__main__":
    print("=" * 60)
    print("Collecting news articles...")
    news = collect_all_news(save=True)
    print(f"\nNews collected: {len(news)} articles")

    print("\n" + "=" * 60)
    print("Collecting crude oil prices...")
    crude = collect_crude_oil(save=True)
    print(f"Crude oil data: {len(crude)} trading days")

    print("\nDone. Files saved to data/raw/")
