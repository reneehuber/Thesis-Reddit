import pandas as pd
from pathlib import Path

# Stage 4: validate the reply structure
# Reddit encodes parent IDs with a type prefix: t1_ means the parent is a comment, t3_ means the parent is a post
# This script strips those prefixes, stores the cleaned ID and the parent type separately, and checks whether each comment's parent actually exists in the dataset
#
# Comments whose parent cannot be matched ("orphans") are dropped -> Arctic Shift archive does not guarantee complete thread coverage —> deleted posts, removed comments, and content predating the earliest archived date 
# Keeping them would introduce phantom nodes in the network, so they are excluded

DATA_FOLDER = Path("DATA Subreddits")
DROP_ORPHANS = True


def load_csv(filepath):
    return pd.read_csv(filepath, dtype=str, engine="python", on_bad_lines="skip")


def validate_reply_structure(df_comments, df_posts, name):
    n0 = len(df_comments)

    # Strip type prefix and store it separately
    df_comments["parent_id_clean"] = df_comments["parent_id"].apply(
        lambda x: x[3:] if isinstance(x, str) and x[:3] in ("t1_", "t3_") else x
    )
    df_comments["parent_type"] = df_comments["parent_id"].apply(
        lambda x: "comment" if isinstance(x, str) and x.startswith("t1_")
        else ("post" if isinstance(x, str) and x.startswith("t3_") else "unknown")
    )

    valid_comment_ids = set(df_comments["id"].dropna())
    valid_post_ids = set(df_posts["id"].dropna()) if df_posts is not None else set()

    # Vectorised parent existence check (faster than row-by-row apply)
    is_comment_reply = df_comments["parent_type"] == "comment"
    is_post_reply = df_comments["parent_type"] == "post"

    parent_exists = (
        (is_comment_reply & df_comments["parent_id_clean"].isin(valid_comment_ids)) |
        (is_post_reply & df_comments["parent_id_clean"].isin(valid_post_ids))
    )

    n_orphans = (~parent_exists).sum()

    if DROP_ORPHANS:
        df_comments = df_comments[parent_exists].copy()

    n1  = len(df_comments)
    pct = (n1 / n0 * 100) if n0 > 0 else 0
    print(f"  {name:<25} {n0:>8,} -> {n1:>8,}  ({pct:.1f}% retained | {n_orphans:,} orphans dropped)")
    return df_comments


comment_files = sorted(DATA_FOLDER.glob("*_comments_clean3.csv"))
post_files = sorted(DATA_FOLDER.glob("*_posts_clean3.csv"))

print(f"Found {len(comment_files)} comment files and {len(post_files)} post files.\n")

print("-- Validating reply structure --")
total_before, total_after = 0, 0

for f in comment_files:
    name = f.name.replace("_comments_clean3.csv", "")
    df_comments = load_csv(f)
    total_before += len(df_comments)

    post_file = DATA_FOLDER / f"{name}_posts_clean3.csv"
    df_posts = load_csv(post_file) if post_file.exists() else None

    df_comments = validate_reply_structure(df_comments, df_posts, name)
    total_after += len(df_comments)

    df_comments.to_csv(
        DATA_FOLDER / f"{name}_comments_clean4.csv",
        index=False, lineterminator="\n"
    )

print(f"\n  Total: {total_before:,} -> {total_after:,} ({total_after/total_before*100:.1f}% retained)")
print("\n[DONE] Saved as *_comments_clean4.csv")
print("       Columns added: parent_id_clean, parent_type")
print("       Next: run 06_deduplicate.py")