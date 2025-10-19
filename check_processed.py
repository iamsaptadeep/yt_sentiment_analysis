# -*- coding: utf-8 -*-
# check_processed.py
import sys
from pathlib import Path
import pandas as pd

p = Path("data/processed")
files = sorted(p.glob("processed_*.csv"))
if not files:
    print("NO processed files found in data/processed")
    sys.exit(1)

f = files[-1]
print("Using file:", f)
try:
    df = pd.read_csv(f, parse_dates=["published"], infer_datetime_format=True)
except Exception as e:
    print("Warning: parse_dates failed:", e)
    df = pd.read_csv(f)

print("\n--- BASIC INFO ---")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

print("\n--- FIRST 8 ROWS ---")
print(df.head(8).to_string(index=False))

print("\n--- PUBLISHED INFO ---")
if "published" in df.columns:
    print("Non-null published count:", df["published"].notna().sum())
    print("published dtype:", df["published"].dtype)
    print("Sample published values:", df["published"].dropna().head(5).tolist())
else:
    print("No 'published' column found.")

print("\n--- LANGUAGE INFO ---")
if "lang" in df.columns:
    vc = df["lang"].fillna("NA").value_counts().head(20)
    print(vc.to_string())
else:
    print("No 'lang' column found.")

print("\n--- SENTIMENT COUNTS ---")
if "sentiment_vader" in df.columns:
    print(df["sentiment_vader"].value_counts(dropna=False).to_string())
else:
    print("No 'sentiment_vader' column found.")
