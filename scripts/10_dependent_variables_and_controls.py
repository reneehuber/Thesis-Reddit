import pandas as pd
import numpy as np
from pathlib import Path

# Computes per-user dependent variables and activity control variables
#
# Dependent variables:
#   - unique_repliers: number of distinct users who directly replied to user i
#     Corresponds to the in-degree of node i in the reply network
#     Measures interaction breadth —> how widely a user's contributions draw responses from distinct community members
#   - mean_score: average net upvote score across all of user i's contributions 
#     Captures community approval of a user's content
#
# Activity controls:
#   - comment_count: total number of comments posted in the subreddit
#   - post_count: total number of posts submitted
#   - out_degree: number of replies user i sent to others (from centrality file)
#   - tenure_days: days between first and last comment in the subreddit, computed from created_utc timestamps
#     Uses comment timestamps rather than account creation dates because the latter had substantial missing values

DATA_FOLDER = Path("DATA Subreddits")
OUTPUT_FOLDER = Path("RESULTS")
OUTPUT_FOLDER.mkdir(exist_ok=True)


def load_csv(filepath):
    return pd.read_csv(filepath, dtype=str, engine="python", on_bad_lines="skip")


comment_files = sorted(DATA_FOLDER.glob("*_comments_final.csv"))
print(f"Found {len(comment_files)} subreddits to process.\n")

all_dv = []

for f in comment_files:
    name = f.name.replace("_comments_final.csv", "")
    print(f"── {name}")

    df = load_csv(f)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # ── Unique repliers ───────────────────────────────────────────────────────
    # For each comment, identify who wrote the parent
    # Then for each user, count how many distinct people replied to them
    id_to_author = df.set_index("id")["author"].to_dict()
    df["parent_author"] = df["parent_id_clean"].map(id_to_author)

    replies = df[
        df["parent_author"].notna() &
        (df["author"] != df["parent_author"])
    ][["author", "parent_author"]].copy()

    unique_repliers = (
        replies.groupby("parent_author")["author"]
        .nunique()
        .reset_index()
        .rename(columns={"parent_author": "author", "author": "unique_repliers"})
    )

    # ── Mean score ────────────────────────────────────────────────────────────
    # Computed from comments only, consistent with the network which is also built exclusively from comment reply relationships
    # Users who only posted without commenting have no network position and are excluded anyway
    score_agg = (
        df.groupby("author")["score"]
        .agg(mean_score="mean", median_score="median",
             total_score="sum", n_comments="count")
        .reset_index()
    )

    # ── Comment count ─────────────────────────────────────────────────────────
    comment_count = df.groupby("author").size().reset_index(name="comment_count")

    # ── Tenure ────────────────────────────────────────────────────────────────
    if "created_utc" in df.columns:
        df["created_utc_num"] = pd.to_numeric(df["created_utc"], errors="coerce")
        tenure_df = (
            df.groupby("author")
            .agg(first_utc=("created_utc_num", "min"),
                 last_utc=("created_utc_num",  "max"))
            .reset_index()
        )
        tenure_df["tenure_days"] = ((tenure_df["last_utc"] - tenure_df["first_utc"]) / 86400).clip(lower=0)
        tenure_df = tenure_df[["author", "tenure_days"]]
    else:
        print(f"  [WARNING] created_utc missing — tenure will be NaN")
        tenure_df = pd.DataFrame(columns=["author", "tenure_days"])

    # ── Post count ────────────────────────────────────────────────────────────
    post_file = DATA_FOLDER / f"{name}_posts_final.csv"
    if post_file.exists():
        post_count = (
            load_csv(post_file).groupby("author")
            .size().reset_index(name="post_count")
        )
    else:
        print(f"  [WARNING] No posts file for {name} — post_count set to 0")
        post_count = pd.DataFrame(columns=["author", "post_count"])

    # ── Merge everything ──────────────────────────────────────────────────────
    users = df[["author"]].drop_duplicates()
    users = users.merge(unique_repliers, on="author", how="left")
    users = users.merge(score_agg, on="author", how="left")
    users = users.merge(comment_count, on="author", how="left")
    users = users.merge(tenure_df, on="author", how="left")
    users = users.merge(post_count, on="author", how="left")

    users["unique_repliers"] = users["unique_repliers"].fillna(0).astype(int)
    users["post_count"] = users["post_count"].fillna(0).astype(int)
    users["comment_count"] = users["comment_count"].fillna(0).astype(int)
    users["subreddit"] = name

    all_dv.append(users)
    print(f"  {len(users):,} users  |  avg unique repliers: {users['unique_repliers'].mean():.2f}"
          f"  |  avg mean score: {users['mean_score'].mean():.2f}")
    print()

df_all = pd.concat(all_dv, ignore_index=True)
df_all.to_csv(OUTPUT_FOLDER / "user_dependent_variables.csv", index=False)

print(f"[DONE] Saved user_dependent_variables.csv")
print(f"       {len(df_all):,} users across {df_all['subreddit'].nunique()} subreddits")
print("       Next: run 11_textual_covariates.py")