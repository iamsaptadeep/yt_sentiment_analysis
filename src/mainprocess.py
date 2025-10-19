# src/fetch_and_process.py
import os
import time
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, DetectorFactory

# Visualization
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Setup
DetectorFactory.seed = 0
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing YOUTUBE_API_KEY in .env")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR / "outputs"
for p in (RAW_DIR, PROC_DIR, OUT_DIR):
    p.mkdir(parents=True, exist_ok=True)


def youtube_client():
    return build("youtube", "v3", developerKey=API_KEY)


def clean_text(t: str) -> str:
    t = str(t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"&amp;|&lt;|&gt;", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def safe_lang(s: str) -> str:
    try:
        return detect(s) if s.strip() else "unknown"
    except Exception:
        return "unknown"

def fetch_comments(video_id: str, max_pages: Optional[int] = None, max_comments: int = 5000) -> List[dict]:
    """
    Fetch comments and replies up to max_comments (default 5000).
    Correctly read the comment id from the API response.
    """
    yt = youtube_client()
    comments: List[dict] = []
    token = None
    page = 0

    while True:
        try:
            resp = yt.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=token,
                textFormat="plainText"
            ).execute()
        except HttpError as e:
            logging.warning(f"HttpError while fetching comments: {e}. Sleeping 5s and retrying once.")
            time.sleep(5)
            try:
                resp = yt.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=token,
                    textFormat="plainText"
                ).execute()
            except Exception as e2:
                logging.error("Second attempt failed: %s", e2)
                break

        items = resp.get("items", [])
        for item in items:
            # top-level comment: id is at item['snippet']['topLevelComment']['id']
            tlc = item.get("snippet", {}).get("topLevelComment", {})
            top_snip = tlc.get("snippet", {})
            top_id = tlc.get("id") or top_snip.get("id")  # fallback

            comments.append({
                "comment_id": top_id,
                "author": top_snip.get("authorDisplayName"),
                "comment": top_snip.get("textDisplay"),
                "likes": top_snip.get("likeCount", 0),
                "published": top_snip.get("publishedAt"),
                "video_id": video_id,
                "parent_id": None,
                "raw": top_snip
            })

            # replies (each reply has its own 'id' sibling to 'snippet')
            if item.get("replies"):
                for r in item["replies"].get("comments", []):
                    rs_snip = r.get("snippet", {})
                    reply_id = r.get("id") or rs_snip.get("id")
                    comments.append({
                        "comment_id": reply_id,
                        "author": rs_snip.get("authorDisplayName"),
                        "comment": rs_snip.get("textDisplay"),
                        "likes": rs_snip.get("likeCount", 0),
                        "published": rs_snip.get("publishedAt"),
                        "video_id": video_id,
                        "parent_id": top_id,
                        "raw": rs_snip
                    })

        page += 1
        logging.info("Fetched page %d, total comments so far: %d", page, len(comments))

        if len(comments) >= max_comments:
            logging.info("Reached limit of %d comments, stopping.", max_comments)
            break

        token = resp.get("nextPageToken")
        if not token or (max_pages and page >= max_pages):
            break

        time.sleep(0.1)

    return comments[:max_comments]



def vader_label(c: float) -> str:
    """
    Strict segmentation:
      Positive: compound >= 0.05
      Negative: compound <= -0.05
      Neutral: otherwise
    """
    if c >= 0.05:
        return "Positive"
    elif c <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def make_wordcloud(text_series, out_path: Path, sentiment_label: str):
    stopwords = set(STOPWORDS)
    stopwords.update(["https", "http", "amp", "video", "youtube", "watch", "com"])
    
    # Preprocess: lowercase and remove short tokens
    texts = [str(t).lower() for t in text_series if isinstance(t, str) and len(str(t).strip()) > 1]
    text = " ".join(texts)
    
    if not text.strip():
        logging.info("No text available for wordcloud %s - skipping.", sentiment_label)
        return

    # Choose colormap based on sentiment
    sentiment = sentiment_label.lower()
    if sentiment == "positive":
        colormap = "Greens"
    elif sentiment == "negative":
        colormap = "Reds"
    else:
        colormap = "Greys"  # fallback / neutral
    
    # Generate wordcloud
    wc = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        stopwords=stopwords,
        colormap=colormap,
        max_words=200
    ).generate(text)
    
    # Save image
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Most Frequent Words in {sentiment_label} Comments", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info("Saved wordcloud: %s", out_path)




def run(video_id: str, max_pages: Optional[int] = None, max_comments: int = 5000) -> Path:
    ts = int(time.time())
    comments = fetch_comments(video_id, max_pages=max_pages, max_comments=max_comments)
    if not comments:
        raise RuntimeError("No comments fetched - check video id or API quota.")
    raw_path = RAW_DIR / f"comments_{video_id}_{ts}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)
    logging.info("Saved raw: %s", raw_path)

    df = pd.DataFrame(comments).drop_duplicates(subset=["comment_id"])
    df["clean_comment"] = df["comment"].fillna("").apply(clean_text)
    df["lang"] = df["clean_comment"].apply(safe_lang)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")

    analyzer = SentimentIntensityAnalyzer()
    scores = df["clean_comment"].apply(lambda x: analyzer.polarity_scores(str(x)))
    scores_df = pd.DataFrame(list(scores))
    df = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
    df["sentiment_vader"] = df["compound"].apply(vader_label)

    out_csv = PROC_DIR / f"processed_{video_id}_{ts}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logging.info("Saved processed: %s", out_csv)

    # quick KPI export for the dashboard to read fast
    kpi = {
        "video_id": video_id,
        "timestamp": datetime.utcnow().isoformat(),
        "n_comments": int(len(df)),
        "pct_negative": float((df["sentiment_vader"] == "Negative").mean() * 100),
        "pct_positive": float((df["sentiment_vader"] == "Positive").mean() * 100),
        "pct_neutral": float((df["sentiment_vader"] == "Neutral").mean() * 100),
        "avg_compound": float(df["compound"].mean())
    }
    pd.DataFrame([kpi]).to_csv(OUT_DIR / f"kpi_{video_id}_{ts}.csv", index=False)
    logging.info("Saved KPI file.")

    # Generate wordclouds for positive and negative (neutral intentionally excluded)
    pos_path = OUT_DIR / f"wordcloud_positive_{video_id}_{ts}.png"
    neg_path = OUT_DIR / f"wordcloud_negative_{video_id}_{ts}.png"
    make_wordcloud(df[df["sentiment_vader"] == "Positive"]["clean_comment"], pos_path, "Positive")
    make_wordcloud(df[df["sentiment_vader"] == "Negative"]["clean_comment"], neg_path, "Negative")

    return out_csv


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True, help="YouTube video id")
    parser.add_argument("--max_pages", type=int, default=None, help="Max pages to fetch")
    parser.add_argument("--max_comments", type=int, default=5000, help="Max comments to fetch (default 5000)")
    args = parser.parse_args()
    run(args.video_id, max_pages=args.max_pages, max_comments=args.max_comments)


