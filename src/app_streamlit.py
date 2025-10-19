# src/app_streamlit.py
import streamlit as st
import pandas as pd
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
OUT = BASE / "data" / "outputs"

st.set_page_config(page_title="YouTube Comments Sentiment", layout="wide")
st.title("YouTube Comments Sentiment Analysis")

# --- Load file safely ---
files = sorted(PROC.glob("processed_*.csv"))
if not files:
    st.error("No processed CSV found. Run fetch_and_process.py first.")
    st.stop()

latest = files[-1]
try:
    df = pd.read_csv(latest, parse_dates=["published"])
except Exception:
    df = pd.read_csv(latest)
    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")

st.markdown(f"**Data file:** `{latest.name}`")
st.write(f"Total rows in file: **{len(df):,}**")

# --- Ensure expected columns exist ---
expected_cols = ["clean_comment", "lang", "sentiment_vader", "likes", "published", "author", "compound"]
for c in expected_cols:
    if c not in df.columns:
        df[c] = np.nan

# Show a small sample for transparency
with st.expander("Show sample rows (first 5)"):
    st.dataframe(df.head(5))

# --- Diagnostics for non-tech users ---
st.write("### Data diagnostics")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Rows total", f"{len(df):,}")
col_b.metric("Non-null published", f"{df['published'].notna().sum():,}")
col_c.metric("Unique languages", f"{df['lang'].nunique():,}")

st.write("Sentiment counts (raw):")
st.write(df["sentiment_vader"].value_counts(dropna=False).to_frame("count"))

# --- Sidebar filters ---
st.sidebar.header("Filters")

langs = ["all"] + sorted([x for x in df["lang"].dropna().unique() if x != "unknown"])
lang = st.sidebar.selectbox("Language", langs, index=0)

# Date handling: use published if available, otherwise fallback window
if df["published"].notna().any():
    min_date = df["published"].min().date()
    max_date = df["published"].max().date()
else:
    from datetime import date, timedelta
    max_date = date.today()
    min_date = max_date - timedelta(days=30)

start_date = st.sidebar.date_input("Start date", value=min_date)
end_date = st.sidebar.date_input("End date", value=max_date)

# Apply filters stepwise and show counts
f = df.copy()
before_filters = len(f)

if lang != "all":
    f = f[f["lang"] == lang]

if f["published"].notna().any():
    f = f[(f["published"].dt.date >= start_date) & (f["published"].dt.date <= end_date)]
else:
    st.warning("Most rows have missing/invalid 'published' timestamps — date filter is skipped.")

after_filters = len(f)
st.write(f"Rows after filters: **{after_filters:,}** (was {before_filters:,})")

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Comments (after filters)", f"{len(f):,}")
col2.metric("Positive %", f"{(f['sentiment_vader']=='Positive').mean()*100:.2f}%")
col3.metric("Neutral %", f"{(f['sentiment_vader']=='Neutral').mean()*100:.2f}%")
col4.metric("Negative %", f"{(f['sentiment_vader']=='Negative').mean()*100:.2f}%")

# --- Charts: Bar + Pie + Time series ---
st.subheader("Sentiment Distribution")

# Build a proper DataFrame for plotting
dist = f["sentiment_vader"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
bar_df = dist.rename_axis("Sentiment").reset_index(name="Count")

# Bar chart using Plotly with custom colors
fig_bar = px.bar(
    bar_df,
    x="Sentiment",
    y="Count",
    title="Sentiment Distribution (Bar)",
    labels={"Count": "Number of Comments"},
    color="Sentiment",  # use sentiment to control bar color
    color_discrete_map={
        "Positive": "green",
        "Neutral": "grey",
        "Negative": "red"
    }
)

# Remove legend (optional, since colors are self-explanatory)
fig_bar.update_layout(showlegend=False)

# Show chart in Streamlit
st.plotly_chart(fig_bar, use_container_width=True)




# Pie chart with custom colors
pie_df = bar_df.set_index("Sentiment").reindex(["Positive","Neutral","Negative"]).reset_index().fillna(0)
color_map = {"Positive":"green", "Neutral":"gray", "Negative":"red"}
fig_pie = px.pie(pie_df, names="Sentiment", values="Count",
                 title="Sentiment Share (Pie)",
                 color="Sentiment",
                 color_discrete_map=color_map)
fig_pie.update_traces(textinfo='percent+label')
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Average Sentiment Over Time (Compound)")
if f["published"].notna().any():
    line = f.set_index("published").resample("D")["compound"].mean().dropna()
    st.line_chart(line)
else:
    st.info("No valid published timestamps to plot time series.")

# --- Top negative comments ---
st.subheader("Top Negative Comments (by Likes)")
neg = f[f["sentiment_vader"] == "Negative"].sort_values("likes", ascending=False)
st.dataframe(neg[["author","clean_comment","likes","published"]].head(50), width=1200)


# --- Wordclouds (show pre-generated if available) ---
st.subheader("Positive vs Negative Wordclouds")
col1, col2 = st.columns(2)
pos_imgs = sorted(OUT.glob("wordcloud_positive_*.png"))
neg_imgs = sorted(OUT.glob("wordcloud_negative_*.png"))
if pos_imgs:
    col1.image(str(pos_imgs[-1]), caption="Positive Comments Wordcloud", width='stretch')
else:
    col1.info("No positive wordcloud available. Run fetch_and_process.py to generate.")

if neg_imgs:
    col2.image(str(neg_imgs[-1]), caption="Negative Comments Wordcloud", width='stretch')
else:
    col2.info("No negative wordcloud available. (May be skipped if few negative comments.)")

# --- Download CSV button for stakeholders ---
st.subheader("Download processed CSV")
st.markdown("Download the filtered CSV for offline review or for sharing with non-technical stakeholders.")
if not f.empty:
    csv_bytes = f.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv_bytes, file_name=f"filtered_comments_{latest.name}", mime="text/csv")
else:
    st.info("No rows to download after filters.")




