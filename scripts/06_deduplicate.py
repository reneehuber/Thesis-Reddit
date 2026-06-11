import pandas as pd
from pathlib import Path

# Stage 5: remove duplicate rows
# Duplicates can appear in the Arctic Shift archive when content is indexed more than once
# The first occurrence of each unique comment/post ID is kept, subsequent ones are dropped
# This is the final preprocessing step —> the *_final.csv files produced here feed directly into network construction and variable computation

DATA_FOLDER = Path("DATA Subreddits")


def load_csv(filepath):
    return pd.read_csv(filepath, dtype=str, engine="python", on_bad_lines="skip")


def deduplicate(df, id_col, name, file_type):
    n0 = len(df)
    df = df.drop_duplicates(subset=[id_col], keep="first")
    dropped = n0 - len(df)
    if dropped > 0:
        print(f"  [WARNING] {name} {file_type}: dropped {dropped:,} duplicate {id_col}s")
    else:
        print(f"  [OK] {name} {file_type}: no duplicates found")
    return df


comment_files = sorted(DATA_FOLDER.glob("*_comments_clean4.csv"))
post_files = sorted(DATA_FOLDER.glob("*_posts_clean3.csv"))

print(f"Found {len(comment_files)} comment files and {len(post_files)} post files.\n")

print("-- Deduplicating --")
for f in comment_files:
    name = f.name.replace("_comments_clean4.csv", "")
    df = load_csv(f)
    df = deduplicate(df, "id", name, "comments")
    df.to_csv(DATA_FOLDER / f"{name}_comments_final.csv", index=False, lineterminator="\n")

print()
for f in post_files:
    name = f.name.replace("_posts_clean3.csv", "")
    df = load_csv(f)
    df = deduplicate(df, "id", name, "posts")
    df.to_csv(DATA_FOLDER / f"{name}_posts_final.csv", index=False, lineterminator="\n")

# Final row count summary with post-reply link check
# The link percentage shows how many comments replying to a post could be
# matched back to a known post ID — a useful sanity check after orphan removal
print("\n-- Final counts --")
rows = []
for f in sorted(DATA_FOLDER.glob("*_comments_final.csv")):
    name = f.name.replace("_comments_final.csv", "")
    df_c = load_csv(f)
    post_file = DATA_FOLDER / f"{name}_posts_final.csv"
    df_p = load_csv(post_file) if post_file.exists() else None
    n_comments = len(df_c)
    n_posts = len(df_p) if df_p is not None else 0

    if df_p is not None and "parent_id_clean" in df_c.columns:
        valid_post_ids = set(df_p["id"].dropna())
        post_replies = df_c[df_c["parent_type"] == "post"]
        linked = post_replies["parent_id_clean"].isin(valid_post_ids).sum()
        link_pct = (linked / len(post_replies) * 100) if len(post_replies) > 0 else 0
        link_note = f"{link_pct:.0f}% of post-replies linked"
    else:
        link_note = "n/a"

    print(f"  {name:<25} {n_comments:>10,} comments   {n_posts:>8,} posts   {link_note}")
    rows.append({"subreddit": name, "comments": n_comments, "posts": n_posts})

summary = pd.DataFrame(rows)
summary.to_csv(DATA_FOLDER / "CLEANING_SUMMARY.csv", index=False)
print(f"\n  TOTAL: {summary['comments'].sum():,} comments, {summary['posts'].sum():,} posts")

print("\n[DONE] Final files saved as *_comments_final.csv and *_posts_final.csv")
print("       Next: run 07_fix_timestamps.py")