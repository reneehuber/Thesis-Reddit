import pandas as pd
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Computes three textual covariates per user:
#
#   - avg_sentiment: average VADER compound score across all of a user's comments
#     VADER is a rule-based sentiment analyser designed for social media text
#     Compound score ranges from -1 (most negative) to +1 (most positive)
#     Included because comment tone may independently affect how other users respond
#
#   - avg_text_length: average word count per comment. Longer contributions provide more material for others to engage with
#
#   - topical_specialisation: proportion of a user's comments that contain at least one keyword from the product-specific dictionary (see script 09)
#     Users whose comments concentrate on product-related content may attract more engagement from others seeking product information
#


DATA_FOLDER = Path("DATA Subreddits")
KEYWORD_FOLDER = Path("KEYWORDS")
OUTPUT_FOLDER = Path("RESULTS")
OUTPUT_FOLDER.mkdir(exist_ok=True)

CATEGORY_MAP = {
    "RobotVacuums": "robot_vacuum",
    "Roborock": "robot_vacuum",
    "roomba": "robot_vacuum",
    "airfryer": "kitchen_appliance",
    "instantpot": "kitchen_appliance",
}


def load_csv(filepath):
    return pd.read_csv(filepath, dtype=str, engine="python", on_bad_lines="skip")


# Load keyword dictionaries
df_robot = pd.read_csv(KEYWORD_FOLDER / "keywords_robot_vacuum.csv")
df_kitchen = pd.read_csv(KEYWORD_FOLDER / "keywords_kitchen_appliance.csv")
KEYWORD_DICT = {
    "robot_vacuum": df_robot["keyword"].str.lower().tolist(),
    "kitchen_appliance": df_kitchen["keyword"].str.lower().tolist(),
}
print(f"Robot vacuum keywords: {len(KEYWORD_DICT['robot_vacuum'])}")
print(f"Kitchen appliance keywords: {len(KEYWORD_DICT['kitchen_appliance'])}\n")

analyser = SentimentIntensityAnalyzer()


def compute_sentiment(text):
    """VADER compound score for a single comment. Returns NaN for empty strings."""
    if not isinstance(text, str) or text.strip() == "":
        return np.nan
    return analyser.polarity_scores(text)["compound"]


def is_topical(text, keywords):
    """Returns 1 if any keyword appears in the comment text, 0 otherwise."""
    if not isinstance(text, str) or text.strip() == "":
        return 0
    text_lower = text.lower()
    return int(any(kw in text_lower for kw in keywords))


def word_count(text):
    if not isinstance(text, str):
        return np.nan
    return len(text.split())


comment_files = sorted(DATA_FOLDER.glob("*_comments_final.csv"))
print(f"Found {len(comment_files)} subreddits to process.\n")

all_ivs = []

for f in comment_files:
    name = f.name.replace("_comments_final.csv", "")
    category = CATEGORY_MAP.get(name)

    if category is None:
        print(f"  [SKIPPED] {name} — no category mapping")
        continue

    keywords = KEYWORD_DICT[category]
    print(f"── {name} ({category})")

    df = load_csv(f)
    print(f"  Loaded {len(df):,} comments")

    # ── Text length ───────────────────────────────────────────────────────────
    df["text_length"] = df["body"].apply(word_count)
    text_length_agg = (
        df.groupby("author")["text_length"]
        .mean().reset_index()
        .rename(columns={"text_length": "avg_text_length"})
    )

    # ── Sentiment ─────────────────────────────────────────────────────────────
    print(f"  Computing sentiment ...")
    df["sentiment"] = df["body"].apply(compute_sentiment)
    sentiment_agg = (
        df.groupby("author")["sentiment"]
        .mean().reset_index()
        .rename(columns={"sentiment": "avg_sentiment"})
    )

    # ── Topical specialisation ────────────────────────────────────────────────
    print(f"  Computing topical specialisation ...")
    df["is_topical"] = df["body"].apply(lambda x: is_topical(x, keywords))
    topical_agg = (
        df.groupby("author")["is_topical"]
        .mean().reset_index()
        .rename(columns={"is_topical": "topical_specialisation"})
    )

    users = df[["author"]].drop_duplicates()
    users = users.merge(text_length_agg, on="author", how="left")
    users = users.merge(sentiment_agg, on="author", how="left")
    users = users.merge(topical_agg, on="author", how="left")
    users["subreddit"] = name

    all_ivs.append(users)
    print(f"  {len(users):,} users  |  avg text length: {users['avg_text_length'].mean():.1f} words"
          f"  |  avg sentiment: {users['avg_sentiment'].mean():.3f}"
          f"  |  avg topical spec: {users['topical_specialisation'].mean():.3f}")
    print()

df_all = pd.concat(all_ivs, ignore_index=True)
df_all.to_csv(OUTPUT_FOLDER / "user_secondary_ivs.csv", index=False)

print(f"[DONE] Saved user_secondary_ivs.csv")
print(f"       {len(df_all):,} users across {df_all['subreddit'].nunique()} subreddits")
print("       Next: run 12_merge.py")