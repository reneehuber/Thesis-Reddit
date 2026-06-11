import pandas as pd
from pathlib import Path

# Stage 2: remove entries that can't be used in the analysis
# This covers deleted/removed accounts, the AutoModerator bot, and any rows missing required fields (author, body, id, parent_id)
# For posts, missing selftext is kept as an empty string —> a title-only post is still valid

DATA_FOLDER = Path("DATA Subreddits")

INVALID_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", "None", "nan"}
INVALID_TEXT = {"[deleted]", "[removed]", "None", "nan", ""}


def load_csv(filepath):
    return pd.read_csv(filepath, dtype=str, engine="python", on_bad_lines="skip")


def clean_comments(df, name):
    n0 = len(df)
    df = df[~df["author"].astype(str).isin(INVALID_AUTHORS)]
    df = df[~df["body"].astype(str).isin(INVALID_TEXT)]
    df = df[df["body"].notna() & df["author"].notna() &
            df["id"].notna() & df["parent_id"].notna()]
    n1  = len(df)
    pct = (n1 / n0 * 100) if n0 > 0 else 0
    print(f"  {name:<25} {n0:>8,} -> {n1:>8,}  ({pct:.1f}% retained)")
    return df


def clean_posts(df, name):
    n0 = len(df)
    df = df[~df["author"].astype(str).isin(INVALID_AUTHORS)]
    df = df[~df["title"].astype(str).isin(INVALID_TEXT)]
    df = df[df["title"].notna() & df["id"].notna()]
    # Replace invalid selftext with empty string instead of dropping the post
    if "selftext" in df.columns:
        df["selftext"] = df["selftext"].astype(str)
        df.loc[df["selftext"].isin(INVALID_TEXT), "selftext"] = ""
    n1  = len(df)
    pct = (n1 / n0 * 100) if n0 > 0 else 0
    print(f"  {name:<25} {n0:>8,} -> {n1:>8,}  ({pct:.1f}% retained)")
    return df


comment_files = sorted(DATA_FOLDER.glob("*_comments_clean.csv"))
post_files = sorted(DATA_FOLDER.glob("*_posts_clean.csv"))

print(f"Found {len(comment_files)} comment files and {len(post_files)} post files.\n")

print("-- Cleaning comments --")
total_before, total_after = 0, 0
for f in comment_files:
    name = f.name.replace("_comments_clean.csv", "")
    df = load_csv(f)
    total_before += len(df)
    df = clean_comments(df, name)
    total_after += len(df)
    df["subreddit_derived"] = name
    df.to_csv(DATA_FOLDER / f"{name}_comments_clean2.csv", index=False, lineterminator="\n")

print(f"\n  Total: {total_before:,} -> {total_after:,} ({total_after/total_before*100:.1f}% retained)\n")

print("-- Cleaning posts --")
total_before, total_after = 0, 0
for f in post_files:
    name = f.name.replace("_posts_clean.csv", "")
    df = load_csv(f)
    total_before += len(df)
    df = clean_posts(df, name)
    total_after += len(df)
    df["subreddit_derived"] = name
    df.to_csv(DATA_FOLDER / f"{name}_posts_clean2.csv", index=False, lineterminator="\n")

print(f"\n  Total: {total_before:,} -> {total_after:,} ({total_after/total_before*100:.1f}% retained)")

print("\n-- Row counts after cleaning --")
rows = []
for f in sorted(DATA_FOLDER.glob("*_comments_clean2.csv")):
    name = f.name.replace("_comments_clean2.csv", "")
    n_comments = len(load_csv(f))
    post_file = DATA_FOLDER / f"{name}_posts_clean2.csv"
    n_posts = len(load_csv(post_file)) if post_file.exists() else 0
    rows.append({"subreddit": name, "comments": n_comments, "posts": n_posts, "total": n_comments + n_posts})
    print(f"  {name:<25} {n_comments:>10,} comments   {n_posts:>8,} posts")
print(f"\n  Grand total: {sum(r['total'] for r in rows):,} rows")

print("\n[DONE] Saved as *_comments_clean2.csv and *_posts_clean2.csv")
print("       Next: run 04_clean_text.py")