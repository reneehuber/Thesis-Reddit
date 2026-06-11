import pandas as pd
from pathlib import Path

# Stage 1 of preprocessing: standardise column formats across all files
# Ensures ID fields are strings, text fields are strings, and tags each file with its subreddit name

DATA_FOLDER = Path("DATA Subreddits")

REQUIRED_COMMENTS = {"id", "author", "body", "created_utc", "parent_id", "score", "subreddit"}
REQUIRED_POSTS = {"id", "author", "title", "created_utc", "score", "subreddit"}


def load_csv(filepath):
    return pd.read_csv(filepath, dtype=str, engine="python", on_bad_lines="skip")


def standardise(df, required_cols, subreddit_name, file_type):
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  [WARNING] {subreddit_name} {file_type} is missing columns: {missing}")
    else:
        print(f"  [OK] {subreddit_name} {file_type} — {len(df):,} rows")

    # Keep created_utc as a raw numeric string —> script 07 patches it from the original JSONL and script 10 converts it to numeric for tenure computation.
    # Converting to datetime here breaks the merge in script 07

    for col in ["id", "parent_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in ["body", "title", "selftext"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["subreddit_derived"] = subreddit_name
    return df


comment_files = sorted(DATA_FOLDER.glob("*_comments_clean.csv"))
post_files = sorted(DATA_FOLDER.glob("*_posts_clean.csv"))

print(f"Found {len(comment_files)} comment files and {len(post_files)} post files.\n")

print("-- Standardising comments --")
for f in comment_files:
    name = f.name.replace("_comments_clean.csv", "")
    df = load_csv(f)
    df = standardise(df, REQUIRED_COMMENTS, name, "comments")
    df.to_csv(f, index=False, lineterminator="\n")

print("\n-- Standardising posts --")
for f in post_files:
    name = f.name.replace("_posts_clean.csv", "")
    df = load_csv(f)
    df = standardise(df, REQUIRED_POSTS, name, "posts")
    df.to_csv(f, index=False, lineterminator="\n")

# Quick consistency check — every subreddit should have both a comments and a posts file
print("\n-- Consistency check --")
comment_subs = {f.name.replace("_comments_clean.csv", "") for f in comment_files}
post_subs = {f.name.replace("_posts_clean.csv",    "") for f in post_files}

if comment_subs == post_subs:
    print(f"[OK] All {len(comment_subs)} subreddits present in both comments and posts.")
else:
    only_comments = comment_subs - post_subs
    only_posts = post_subs - comment_subs
    if only_comments:
        print(f"[WARNING] Subreddits with comments but no posts: {only_comments}")
    if only_posts:
        print(f"[WARNING] Subreddits with posts but no comments: {only_posts}")

print("\n[DONE] All files standardised and saved in place.")
print("       Next: run 03_remove_invalid_entries.py")

# Row count summary — useful to verify your numbers match the thesis
print("\n-- Row counts per subreddit --")
rows = []
for f in comment_files:
    name = f.name.replace("_comments_clean.csv", "")
    n_comments = len(load_csv(f))
    post_file = DATA_FOLDER / f"{name}_posts_clean.csv"
    n_posts = len(load_csv(post_file)) if post_file.exists() else 0
    rows.append({"subreddit": name, "comments": n_comments,
                 "posts": n_posts, "total": n_comments + n_posts})

summary = pd.DataFrame(rows).set_index("subreddit")
print(summary.to_string())
print(f"\nGrand total: {summary['total'].sum():,} rows across {len(rows)} subreddits")