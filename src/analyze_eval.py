from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
OUT = BASE / "data" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def load_latest_processed() -> pd.DataFrame:
    files = sorted(PROC.glob("processed_*.csv"))
    if not files:
        raise FileNotFoundError("No processed CSV found. Run fetch_and_process.py first.")
    return pd.read_csv(files[-1], parse_dates=["published"])

def plot_sentiment_distribution(df: pd.DataFrame):
    counts = df["sentiment_vader"].value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0)
    ax = counts.plot(kind="bar", title="Sentiment Distribution", rot=0)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    plt.tight_layout()
    out = OUT / "sentiment_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)

def plot_sentiment_over_time(df: pd.DataFrame):
    s = df.set_index("published").resample("D")["compound"].mean()
    ax = s.plot(title="Average Sentiment (Compound) Over Time")
    plt.tight_layout()
    out = OUT / "sentiment_over_time.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)

def top_negative_table(df: pd.DataFrame):
    neg = df[df["sentiment_vader"]=="Negative"].sort_values("likes", ascending=False)
    cols = ["author","clean_comment","likes","published"]
    out_csv = OUT / "top_negative_comments.csv"
    neg[cols].head(50).to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    df = load_latest_processed()
    plot_sentiment_distribution(df)
    plot_sentiment_over_time(df)
    top_negative_table(df)


