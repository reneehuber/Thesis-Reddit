import pandas as pd
import unicodedata
import re
from pathlib import Path

# Stage 3: text-level cleaning
# Applies Unicode normalisation (NFKC) to standardise character representations —> this handles curly quotes, dashes, zero-width spaces, and similar artefacts that appear frequently in Reddit text + collapses whitespace and line breaks
# Comments shorter than MIN_WORD_COUNT words are dropped as too brief to be analytically useful
# Posts are not filtered by word count —> a short title is valid

DATA_FOLDER = Path("DATA Subreddits")
MIN_WORD_COUNT = 3


def load_csv(filepath):
    return pd.read_csv(filepath, dtype=str, engine="python", on_bad_lines="skip")


def clean_text(text):
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)  # invisible characters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text):
    if not isinstance(text, str):
        return 0
    return len(text.split())


def clean_comments_df(df, name):
    n0 = len(df)
    df["body"] = df["body"].apply(clean_text)
    df = df[df["body"].apply(word_count) >= MIN_WORD_COUNT]
    n1 = len(df)
    pct = (n1 / n0 * 100) if n0 > 0 else 0
    print(f"  {name:<25} {n0:>8,} -> {n1:>8,}  ({pct:.1f}% retained)")
    return df


def clean_posts_df(df, name):
    df["title"] = df["title"].apply(clean_text)
    if "selftext" in df.columns:
        df["selftext"] = df["selftext"].apply(clean_text)
    print(f"  {name:<25} {len(df):>8,} rows  (text cleaned, no rows dropped)")
    return df


comment_files = sorted(DATA_FOLDER.glob("*_comments_clean2.csv"))
post_files = sorted(DATA_FOLDER.glob("*_posts_clean2.csv"))

print(f"Found {len(comment_files)} comment files and {len(post_files)} post files.\n")

print("-- Cleaning comment text --")
total_before, total_after = 0, 0
for f in comment_files:
    name = f.name.replace("_comments_clean2.csv", "")
    df = load_csv(f)
    total_before += len(df)
    df = clean_comments_df(df, name)
    total_after  += len(df)
    df.to_csv(DATA_FOLDER / f"{name}_comments_clean3.csv", index=False, lineterminator="\n")

print(f"\n  Total: {total_before:,} -> {total_after:,} ({total_after/total_before*100:.1f}% retained)\n")

print("-- Cleaning post text --")
for f in post_files:
    name = f.name.replace("_posts_clean2.csv", "")
    df = load_csv(f)
    df = clean_posts_df(df, name)
    df.to_csv(DATA_FOLDER / f"{name}_posts_clean3.csv", index=False, lineterminator="\n")

print("\n-- Row counts after text cleaning --")
for f in sorted(DATA_FOLDER.glob("*_comments_clean3.csv")):
    name = f.name.replace("_comments_clean3.csv", "")
    n = len(load_csv(f))
    print(f"  {name:<25} {n:>10,} comments")

print("\n[DONE] Saved as *_comments_clean3.csv and *_posts_clean3.csv")
print("       Next: run 05_validate_replies.py")