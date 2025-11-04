# app/streamlit_app.py
"""
Streamlit modern-card UI for the News Recommender project.

Features:
- Displays news articles as modern cards
- "Read" opens article in a new tab
- "Like" (thumbs up) stores liked articles in a local SQLite DB via SQLAlchemy
- Build user profile from liked articles (mean of SBERT embeddings)
- Query FAISS index for recommendations
- Excludes already liked articles from recommendations

Instructions:
- Run `python src/embed_and_index.py` before using this app to ensure embeddings/faiss/meta exist.
- Run: `streamlit run app/streamlit_app.py`
"""

from pathlib import Path
import sqlite3
import time
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

# caching / resources
from functools import lru_cache

# embedding & vector search
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# SQLAlchemy for SQLite (safer interface)
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Table, MetaData
from sqlalchemy.orm import sessionmaker

# ---------- Config ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "articles.csv"
FAISS_BIN = PROJECT_ROOT / "embeddings" / "faiss_index.bin"
META_PKL = PROJECT_ROOT / "embeddings" / "meta.pkl"
DB_FILE = PROJECT_ROOT / "history.db"     # SQLite DB at project root
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10  # how many recommendations to show

# ---------- Helpers / Resource Loaders ----------
st.set_page_config(page_title="News Recommender", layout="wide")

@st.cache_resource(show_spinner=False)
def load_model():
    model = SentenceTransformer(MODEL_NAME)
    return model

@st.cache_resource(show_spinner=False)
def load_faiss_and_meta():
    if not FAISS_BIN.exists() or not META_PKL.exists():
        raise FileNotFoundError("FAISS index or meta.pkl not found. Run src/embed_and_index.py first.")
    index = faiss.read_index(str(FAISS_BIN))
    with open(META_PKL, "rb") as f:
        meta_list = pickle.load(f)
    return index, meta_list

@st.cache_data(show_spinner=False)
def load_articles_df():
    if not DATA_FILE.exists():
        raise FileNotFoundError("data/articles.csv not found. Run src/news_fetcher.py first.")
    df = pd.read_csv(DATA_FILE)
    # build a text field for embedding when needed
    df["text_for_embedding"] = (df.get("title", "").fillna("") + ". " +
                                df.get("description", "").fillna("") + ". " +
                                df.get("content", "").fillna(""))
    # ensure urls are str
    df["url"] = df["url"].astype(str)
    return df

# Database helpers (SQLAlchemy)
def get_db_engine():
    engine = create_engine(f"sqlite:///{DB_FILE}")
    return engine

def init_db():
    engine = get_db_engine()
    meta = MetaData()
    likes = Table(
        "likes", meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("url", String, unique=True, nullable=False),
        Column("title", String),
        Column("source_name", String),
        Column("saved_at", DateTime),
    )
    meta.create_all(engine)
    return engine, likes

def get_session():
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    return Session()

# ---------- Core logic ----------
model = load_model()
try:
    faiss_index, meta_list = load_faiss_and_meta()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

articles_df = load_articles_df()
engine, likes_table = init_db()

# utility: normalize vector for cosine search using IndexFlatIP
def normalize_vec(v: np.ndarray):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

@st.cache_data(show_spinner=False)
def embed_texts(texts):
    # returns numpy array (n, dim)
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embs = embs / norms
    return embs

def save_like_to_db(url, title, source_name):
    session = get_session()
    # raw SQLAlchemy Core insert
    conn = engine.connect()
    now = datetime.utcnow()
    try:
        ins = likes_table.insert().values(url=url, title=title, source_name=source_name, saved_at=now)
        conn.execute(ins)
        conn.close()
        return True
    except Exception as e:
        # likely unique constraint violation if already liked
        conn.close()
        return False

def remove_like_from_db(url):
    conn = engine.connect()
    try:
        delete = likes_table.delete().where(likes_table.c.url == url)
        conn.execute(delete)
        conn.close()
        return True
    except Exception:
        conn.close()
        return False

def get_liked_urls():
    conn = engine.connect()
    sel = likes_table.select()
    res = conn.execute(sel).fetchall()
    conn.close()
    return [r["url"] for r in res]

def get_likes_count():
    conn = engine.connect()
    sel = likes_table.select()
    res = conn.execute(sel).fetchall()
    conn.close()
    return len(res)

# compute recommendations given liked URLs
def recommend_for_user(liked_urls, top_k=TOP_K):
    if not liked_urls:
        return []
    # find rows in articles_df matching liked_urls
    liked_rows = articles_df[articles_df["url"].isin(liked_urls)].reset_index(drop=True)
    if liked_rows.empty:
        return []
    texts = liked_rows["text_for_embedding"].tolist()
    emb = embed_texts(texts)  # (n_liked, dim)
    user_vec = np.mean(emb, axis=0, keepdims=True)  # (1, dim)
    user_vec = normalize_vec(user_vec)
    # search faiss
    D, I = faiss_index.search(user_vec.astype(np.float32), top_k * 3)  # retrieve more to filter liked
    idxs = I[0].tolist()
    # meta_list maps faiss index positions -> metadata containing "index" which references row index in original df used when building index
    recs = []
    for pos in idxs:
        if pos < 0 or pos >= len(meta_list):
            continue
        meta = meta_list[pos]
        url = str(meta.get("url"))
        if url in liked_urls:
            continue
        recs.append(meta)
        if len(recs) >= top_k:
            break
    return recs

# ---------- UI layout ----------
st.title("📰 Personalized News Recommender — Geopolitics · Business · Science · AI")
st.markdown("Modern cards UI • Thumbs-up to like • Recommendations built from your liked articles")

# Sidebar: stats & actions
with st.sidebar:
    st.header("Controls")
    if st.button("Refresh recommendations"):
        st.rerun()
    liked_count = get_likes_count()
    st.markdown(f"**Liked articles:** {liked_count}")
    st.markdown("---")
    st.markdown("**Index / Data Info**")
    st.write(f"Articles: {len(articles_df)}")
    st.write(f"FAISS entries: {len(meta_list)}")
    st.markdown("---")
    st.markdown("Run pipeline if you need to update:")
    st.code("python src/news_fetcher.py\npython src/embed_and_index.py", language="bash")

# Main columns: Left feed, Right recommendations
left_col, right_col = st.columns([2, 1])

# --- Left: feed listing (paginated)
PER_PAGE = 8
if "page" not in st.session_state:
    st.session_state.page = 0

def show_article_card(row, key_prefix=""):
    title = row.get("title", "") or "No title"
    source = row.get("source_name", "") or ""
    published = row.get("publishedAt", "") or ""
    description = row.get("description", "") or ""
    url = row.get("url", "")
    # simple card with markdown and buttons
    card_md = f"**{title}**  \n\n_{source}_ • {published}  \n\n{description or ''}"
    st.markdown(card_md)
    cols = st.columns([1, 1, 2])
    # Read button opens in new tab
    read_key = f"read_{key_prefix}_{hash(url)}"
    if cols[0].button("Read ▶️", key=read_key):
        st.experimental_set_query_params(open=url)  # allow linking but also open new tab using JS not available; we provide link
        st.write(f"[Open article in new tab]({url})")
    # Like button
    like_key = f"like_{key_prefix}_{hash(url)}"
    if cols[1].button("👍 Like", key=like_key):
        saved = save_like_to_db(url=url, title=title, source_name=source)
        if saved:
            st.success("Saved to Likes ❤️")
        else:
            st.info("Already liked or error.")
    # show source link on right
    cols[2].write(f"[Source Link]({url})")

# pagination controls
start = st.session_state.page * PER_PAGE
end = start + PER_PAGE
subset = articles_df.iloc[start:end].to_dict(orient="records")

with left_col:
    st.subheader("Latest Feed")
    for i, row in enumerate(subset):
        show_article_card(row, key_prefix=f"feed_{start+i}")
        st.markdown("---")
    # paging buttons
    cols = st.columns([1, 1, 1])
    if cols[0].button("◀ Prev") and st.session_state.page > 0:
        st.session_state.page -= 1
        st.experimental_rerun()
    if cols[1].button("Next ▶") and end < len(articles_df):
        st.session_state.page += 1
        st.experimental_rerun()
    cols[2].write(f"Page {st.session_state.page + 1}")

# --- Right: Recommendations & Liked list
with right_col:
    st.subheader("Your Recommendations")
    liked_urls = get_liked_urls()
    if not liked_urls:
        st.info("You have not liked any articles yet. Like articles (👍) from the feed to get personalized recommendations.")
    else:
        if st.button("Generate Recommendations"):
            with st.spinner("Computing recommendations..."):
                recs = recommend_for_user(liked_urls, top_k=TOP_K)
                if not recs:
                    st.warning("No recommendations found. Try liking more articles or rebuild embeddings.")
                else:
                    for r in recs:
                        title = r.get("title")
                        src = r.get("source_name")
                        pub = r.get("publishedAt")
                        desc = r.get("description") or ""
                        url = r.get("url")
                        st.markdown(f"**{title}**  \n\n_{src}_ • {pub}  \n\n{desc}")
                        c1, c2 = st.columns([1, 1])
                        if c1.button("Read ▶️", key=f"rec_read_{hash(url)}"):
                            st.write(f"[Open article in new tab]({url})")
                        if c2.button("👍 Like", key=f"rec_like_{hash(url)}"):
                            saved = save_like_to_db(url=url, title=title, source_name=src)
                            if saved:
                                st.success("Saved to Likes ❤️")
                            else:
                                st.info("Already liked or error.")
                        st.markdown("---")
    st.markdown("### Liked Articles")
    conn = engine.connect()
    sel = likes_table.select()
    res = conn.execute(sel).fetchall()
    conn.close()
    if not res:
        st.write("No liked articles yet.")
    else:
        for r in res[::-1]:  # show latest first
            st.markdown(f"* {r['title']} — _{r['source_name']}_")
            if st.button("Remove", key=f"remove_{r['url']}"):
                remove_like_from_db(r["url"])
                st.experimental_rerun()

# Footer / notes
st.markdown("---")
st.markdown("Project: Personalized News Recommender — built with SBERT & FAISS. Cards UI • SQLite for likes.")
st.caption("Tip: To update the dataset, run the fetcher & embedding scripts in your terminal, then refresh this app.")

