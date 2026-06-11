import json
import pandas as pd
from pathlib import Path

# Converts the raw JSONL downloads from Arctic Shift into CSV files
# Produces one comments CSV and one posts CSV per subreddit
# These are the starting point for all subsequent preprocessing steps

DATA_FOLDER = Path("DATA Subreddits")

# Maps each JSONL stem to the subreddit name used throughout the rest of the pipeline
# Format: (comments_jsonl_stem, posts_jsonl_stem) -> subreddit label
SUBREDDITS = {
    "airfryer": ("r_Airfryer_comments", "r_Airfryer_posts"),
    "instantpot": ("r_instantpot_comments", "r_instantpot_posts"),
    "Roborock": ("r_Roborock_comments", "r_Roborock_posts"),
    "RobotVacuums": ("r_RobotVacuums_comments", "r_RobotVacuums_posts"),
    "roomba": ("r_roomba_comments", "r_roomba_posts"),
}

# Fields we extract from each JSONL record
COMMENT_FIELDS = ["id", "author", "parent_id", "created_utc", "body", "score", "subreddit"]
POST_FIELDS = ["id", "author", "title", "selftext", "created_utc", "score", "subreddit"]


def jsonl_to_df(filepath, fields):
    """Reads a JSONL file and returns a dataframe with the requested fields."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append({field: obj.get(field, None) for field in fields})
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


print("Converting JSONL files to CSV ...\n")

for subreddit, (comments_stem, posts_stem) in SUBREDDITS.items():
    print(f"── {subreddit}")

    # Comments
    comments_path = DATA_FOLDER / f"{comments_stem}.jsonl"
    if comments_path.exists():
        df_comments = jsonl_to_df(comments_path, COMMENT_FIELDS)
        out = DATA_FOLDER / f"{subreddit}_comments_clean.csv"
        df_comments.to_csv(out, index=False, lineterminator="\n")
        print(f"  Comments: {len(df_comments):,} rows → {out.name}")
    else:
        print(f"  [WARNING] {comments_path.name} not found — skipping")

    # Posts / submissions
    posts_path = DATA_FOLDER / f"{posts_stem}.jsonl"
    if posts_path.exists():
        df_posts = jsonl_to_df(posts_path, POST_FIELDS)
        out = DATA_FOLDER / f"{subreddit}_posts_clean.csv"
        df_posts.to_csv(out, index=False, lineterminator="\n")
        print(f"  Posts:    {len(df_posts):,} rows → {out.name}")
    else:
        print(f"  [WARNING] {posts_path.name} not found — skipping")

    print()

print("[DONE] CSV files saved to DATA Subreddits/")
print("       Next: run 02_precleaning.py")