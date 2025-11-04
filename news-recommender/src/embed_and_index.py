# src/embed_and_index.py
"""
Embed articles and create a FAISS index.

Produces:
- embeddings/faiss_index.bin
- embeddings/meta.pkl  (list of metadata dicts corresponding to index order)
- updates data/articles.csv with a column 'embedding_done' if desired
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# paths
DATA_DIR = Path("data")
EMB_DIR = Path("embeddings")
DATA_FILE = DATA_DIR / "articles.csv"
EMB_DIR.mkdir(parents=True, exist_ok=True)
FAISS_BIN = EMB_DIR / "faiss_index.bin"
META_PKL = EMB_DIR / "meta.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_articles(min_rows=1):
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found. Run news_fetcher first.")
    df = pd.read_csv(DATA_FILE)
    if df.shape[0] < min_rows:
        raise ValueError("Not enough articles to embed.")
    # drop rows without content/title
    df["text_for_embedding"] = (df["title"].fillna("") + ". " + df["description"].fillna("") + ". " + df["content"].fillna(""))
    df = df[df["text_for_embedding"].str.strip() != ""].reset_index(drop=True)
    return df

def embed_texts(texts, model, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    return embeddings

def normalize_embeddings(embs):
    # normalize for cosine similarity (inner product on normalized vectors)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return embs / norms

def build_faiss_index(embs):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors -> cosine
    index.add(embs.astype(np.float32))
    return index

def save_index_and_meta(index, meta_list):
    # save faiss
    faiss.write_index(index, str(FAISS_BIN))
    # save metadata list
    with open(META_PKL, "wb") as f:
        pickle.dump(meta_list, f)
    print(f"Saved FAISS index to {FAISS_BIN} and metadata to {META_PKL}")

def main(rebuild_all=False):
    df = load_articles()
    model = SentenceTransformer(MODEL_NAME)
    texts = df["text_for_embedding"].tolist()

    print(f"Embedding {len(texts)} articles with model {MODEL_NAME}")
    embs = embed_texts(texts, model)
    embs = normalize_embeddings(embs)

    print("Building FAISS index")
    index = build_faiss_index(embs)

    # Build metadata list (so we can map index positions back to article rows)
    meta_list = []
    for idx, row in df.iterrows():
        meta = {
            "index": int(idx),
            "title": row.get("title"),
            "url": row.get("url"),
            "source_name": row.get("source_name"),
            "publishedAt": row.get("publishedAt"),
            "description": row.get("description"),
            "category_hint": row.get("category_hint")
        }
        meta_list.append(meta)

    save_index_and_meta(index, meta_list)
    print("Done.")

if __name__ == "__main__":
    main()
