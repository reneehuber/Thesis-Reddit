import json
import pandas as pd
from pathlib import Path

# Patches created_utc timestamps back into the final comment files using the original JSONL source data
# This step is needed because timestamp fields were lost or malformed during CSV conversion 
# The tenure variable computed in script 10 depends on created_utc being fully populated
#
# If JSONL files are complete, this script will detect that timestamps are already present and skip those files automatically

DATA_FOLDER = Path("DATA Subreddits")

JSONL_TO_SUBREDDIT = {
    "r_Airfryer_comments": "airfryer",
    "r_instantpot_comments": "instantpot",
    "r_Roborock_comments": "Roborock",
    "r_RobotVacuums_comments": "RobotVacuums",
    "r_roomba_comments": "roomba",
}

EXTRACT_FIELDS = {"id", "created_utc"}


def load_csv(filepath):
    return pd.read_csv(filepath, dtype=str, engine="python", on_bad_lines="skip")


def extract_timestamps(jsonl_path):
    """Reads only the id and created_utc fields from a JSONL file —> memory efficient"""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append({field: obj.get(field) for field in EXTRACT_FIELDS})
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


print("Patching timestamps ...\n")

for jsonl_stem, subreddit in JSONL_TO_SUBREDDIT.items():
    jsonl_path = DATA_FOLDER / f"{jsonl_stem}.jsonl"
    final_csv = DATA_FOLDER / f"{subreddit}_comments_final.csv"

    print(f"── {subreddit}")

    if not jsonl_path.exists():
        print(f"  [SKIPPED] {jsonl_path.name} not found\n")
        continue
    if not final_csv.exists():
        print(f"  [SKIPPED] {final_csv.name} not found\n")
        continue

    df = load_csv(final_csv)

    # Check whether timestamps are already populated —> skip if so
    if "created_utc" in df.columns:
        valid = df["created_utc"].dropna()
        valid = valid[~valid.astype(str).str.strip().isin(["", "nan", "None"])]
        if len(valid) / len(df) > 0.9:
            print(f"  [OK] created_utc already populated ({len(valid):,} / {len(df):,} rows) — skipping\n")
            continue

    print(f"  Extracting timestamps from {jsonl_path.name} ...")
    df_ts = extract_timestamps(jsonl_path)
    df_ts["id"] = df_ts["id"].astype(str)
    populated = df_ts["created_utc"].notna().sum()
    print(f"  Extracted {len(df_ts):,} rows — {populated:,} have created_utc")

    # Drop the old (empty) column and merge fresh timestamps in
    df = df.drop(columns=["created_utc"], errors="ignore")
    df = df.merge(df_ts[["id", "created_utc"]], on="id", how="left")

    matched = df["created_utc"].notna().sum()
    match_pct = matched / len(df) * 100
    print(f"  Matched {matched:,} / {len(df):,} rows ({match_pct:.1f}%)")

    if match_pct < 50:
        print(f"  [WARNING] Less than 50% matched — check ID consistency")

    df.to_csv(final_csv, index=False, lineterminator="\n")
    print(f"  Saved -> {final_csv.name}\n")


# ── Verification: check timestamp coverage per subreddit ─────────────────────
# Confirms that created_utc is populated for enough users to make tenure computation meaningful

print("=" * 55)
print("Timestamp coverage check")
print("=" * 55)

for subreddit in JSONL_TO_SUBREDDIT.values():
    f = DATA_FOLDER / f"{subreddit}_comments_final.csv"
    if not f.exists():
        continue
    df = load_csv(f)
    total_users = df["author"].nunique()
    has_utc = df[
        df["created_utc"].notna() &
        (~df["created_utc"].astype(str).str.strip().isin(["nan", "None", ""]))
    ]["author"].nunique()
    pct = has_utc / total_users * 100 if total_users > 0 else 0
    print(f"  {subreddit:<20} {has_utc:>8,} / {total_users:>8,} users have created_utc ({pct:.1f}%)")

print("\n[DONE] Timestamps patched and verified.")
print("       Next: run 08_network_and_centrality.py")