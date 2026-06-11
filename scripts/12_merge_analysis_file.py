import pandas as pd
import numpy as np
from pathlib import Path

# Merges centrality scores, dependent variables, and textual covariates into a single analysis-ready file per subreddit + a combined file for all subreddits together
#
# The merge is performed on author + subreddit
# Users who appear in the centrality file but not in the DVs file (or vice versa) are dropped 
# -> this happens for a small number of users who made only post-level contributions with no comment replies, or whose activity fell below the minimum thresholds in earlier steps

NETWORK_FOLDER = Path("NETWORKS")
RESULTS_FOLDER = Path("RESULTS")
OUTPUT_FOLDER = Path("ANALYSIS")
OUTPUT_FOLDER.mkdir(exist_ok=True)

NUMERIC_COLS = [
    "betweenness", "eigenvector", "pagerank", "in_degree", "out_degree",
    "unique_repliers", "mean_score", "median_score", "total_score",
    "comment_count", "post_count", "tenure_days",
    "avg_text_length", "avg_sentiment", "topical_specialisation",
]

print("Loading input files ...")
df_centrality = pd.read_csv(NETWORK_FOLDER / "ALL_centrality.csv")
df_dvs = pd.read_csv(RESULTS_FOLDER / "user_dependent_variables.csv")
df_ivs = pd.read_csv(RESULTS_FOLDER / "user_secondary_ivs.csv")

print(f"  Centrality : {len(df_centrality):,} rows")
print(f"  DVs : {len(df_dvs):,} rows")
print(f"  Textual IVs: {len(df_ivs):,} rows\n")

print("Merging ...")
df = (
    df_centrality
    .merge(df_dvs, on=["author", "subreddit"], how="inner")
    .merge(
        df_ivs[["author", "subreddit", "avg_text_length", "avg_sentiment", "topical_specialisation"]],
        on=["author", "subreddit"], how="inner"
    )
)

print(f"  Rows after merge : {len(df):,}")
print(f"  Users in centrality file : {len(df_centrality):,}")
print(f"  Users in DVs file : {len(df_dvs):,}")
print(f"  Users dropped in merge : {len(df_centrality) + len(df_dvs) - 2*len(df):,}  (not present in all three files)\n")

for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Descriptive statistics ────────────────────────────────────────────────────

desc_cols = [
    "betweenness", "eigenvector", "pagerank",
    "unique_repliers", "mean_score",
    "comment_count", "post_count", "tenure_days",
    "avg_text_length", "avg_sentiment", "topical_specialisation",
]
desc = df[desc_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
desc.columns = ["n", "mean", "sd", "min", "median", "max"]
print("Descriptive statistics (full sample):")
print(desc.round(4).to_string())

# ── Per-subreddit summary ────────────────────────────────────────────────────

print("\nPer-subreddit averages:")
print(df.groupby("subreddit")[["unique_repliers", "mean_score",
                                "betweenness", "pagerank"]].mean().round(4).to_string())

# ── Missing value check ───────────────────────────────────────────────────────

missing = df[NUMERIC_COLS].isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("\n[OK] No missing values in numeric columns.")
else:
    print("\nMissing values:")
    print(missing.to_string())
    print("  (these rows will be dropped listwise in regression)")

# ── Unique user count across the full dataset ─────────────────────────────────

total_unique_users = df["author"].nunique()
print(f"\nTotal unique users across all subreddits: {total_unique_users:,}")
print("Per subreddit:")
for sub, grp in df.groupby("subreddit"):
    print(f"  {sub:<20} {grp['author'].nunique():>8,} unique users")

# ── Save files ────────────────────────────────────────────────────────────────

for sub in sorted(df["subreddit"].unique()):
    sub_df = df[df["subreddit"] == sub].copy()
    sub_df.to_csv(OUTPUT_FOLDER / f"{sub}_analysis.csv", index=False)
    print(f"  Saved {sub}_analysis.csv ({len(sub_df):,} users)")

df.to_csv(OUTPUT_FOLDER / "ALL_analysis.csv", index=False)
print(f"\n  Saved ALL_analysis.csv ({len(df):,} users across {df['subreddit'].nunique()} subreddits)")

print("\n[DONE] Merge complete. Files in ANALYSIS/")
print("       Review descriptive statistics above, then run 13_regressions.py")