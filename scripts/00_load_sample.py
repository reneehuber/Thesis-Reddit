import json
from pathlib import Path
import pandas as pd

# Sample check of the data format and content before processing everything -> Verify that all the data necessary for the analysis is present
# Reads the first N lines of one JSONL file, extracts key fields, and prints summary stats

# Swap COMMENTS_PATH to test different files if needed
COMMENTS_PATH = Path("DATA Subreddits/r_Airfryer_comments.jsonl")

N = 5000  

rows = []
with COMMENTS_PATH.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= N:
            break
        d = json.loads(line)
        rows.append({
            "id": d.get("id"),
            "author": d.get("author"),
            "parent_id": d.get("parent_id"),
            "created_utc": d.get("created_utc"),
            "body": d.get("body"),
            "subreddit": d.get("subreddit"),
            "score": d.get("score"),
        })

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")

print(df.head(3))
print("\nColumns:", df.columns.tolist())
print("Rows:", len(df))
print("Date range:", df["date"].min(), "→", df["date"].max())
print("Unique authors:", df["author"].nunique())