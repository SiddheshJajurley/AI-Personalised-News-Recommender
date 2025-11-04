import os
import csv
import time
from datetime import datetime
from dotenv import load_dotenv
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv('NEWSAPI_KEY')
if not API_KEY:
    raise ValueError("Please set NEWSAPI_KEY in your .env file")

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = DATA_DIR / "articles.csv"

CATEGORIES = ["business", "science", "technology"]

KEYWORD_QUERIES = [
    "geopolitics OR geopolitical",
    "AI OR \"artificial intelligence\" OR machine learning"
]

BASE_URL = "https://newsapi.org/v2/top-headlines"
EVERYTHING_URL = "https://newsapi.org/v2/everything"

HEADERS = {"Authorization": API_KEY}

def fetch_top_headline(category,page_size = 100):
    """Fetch top headlines for a category (country=en)."""
    params = {
        "category": category,
        "language": "en",
        "pageSize": page_size,
    }
    resp = requests.get(BASE_URL, params={**params, "apiKey": API_KEY})
    resp.raise_for_status()
    data = resp.json()
    return data.get("articles", [])

def fetch_keyword_articles(query, page_size=100):
    """Fetch articles matching a keyword query via everything endpoint."""
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }
    resp = requests.get(EVERYTHING_URL, params={**params, "apiKey": API_KEY})
    resp.raise_for_status()
    data = resp.json()
    return data.get("articles", [])

def normalize_article(a, category_hint=None):
    """Return a normalized dict for CSV storage."""
    return {
        "source_id": a.get("source", {}).get("id"),
        "source_name": a.get("source", {}).get("name"),
        "author": a.get("author"),
        "title": a.get("title"),
        "description": a.get("description"),
        "url": a.get("url"),
        "urlToImage": a.get("urlToImage"),
        "publishedAt": a.get("publishedAt"),
        "content": a.get("content"),
        "category_hint": category_hint,
        "fetched_at": datetime.utcnow().isoformat()
    }

def load_existing_urls():
    if not OUT_FILE.exists():
        return set()
    df = pd.read_csv(OUT_FILE)
    return set(df["url"].dropna().astype(str).tolist())

def save_articles(list_of_dicts):
    """Append new articles to CSV (deduplicated by URL)."""
    if not list_of_dicts:
        print("No new articles to save.")
        return

    fieldnames = [
        "source_id", "source_name", "author", "title",
        "description", "url", "publishedAt", "content",
        "category_hint", "fetched_at"
    ]
    
    existing = load_existing_urls()
    to_write = [a for a in list_of_dicts if a["url"] not in existing]

    if not to_write:
        print("No new unique articles found after dedupe.")
        return

    write_header = not OUT_FILE.exists()
    with OUT_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in to_write:
            writer.writerow(r)
    print(f"Saved {len(to_write)} new articles to {OUT_FILE}")
    
def main():
    all_articles = []

    # 1) category-based fetches
    for cat in CATEGORIES:
        try:
            print(f"Fetching top headlines for category: {cat}")
            articles = fetch_top_headlines(cat)
            for a in articles:
                all_articles.append(normalize_article(a, category_hint=cat))
            time.sleep(1)  # be nice to API
        except Exception as e:
            print(f"Warning: failed to fetch category {cat}: {e}")

    # 2) keyword-based fetches (geopolitics, AI)
    for q in KEYWORD_QUERIES:
        try:
            print(f"Fetching keyword query: {q}")
            articles = fetch_keyword_articles(q)
            for a in articles:
                all_articles.append(normalize_article(a, category_hint="keyword"))
            time.sleep(1)
        except Exception as e:
            print(f"Warning: failed to fetch keyword query {q}: {e}")

    # Save deduplicated
    save_articles(all_articles)
    print("Done fetching.")

# ---- MAIN ENTRY ----
if __name__ == "__main__":
    main()
