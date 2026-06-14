"""
Tests whether the NEGATIVE betweenness coefficient in Model 3 is a genuine
suppression effect or an artifact of multicollinearity with out_degree (r=0.77).

Starting from the M2 base (centrality + dummies + interactions), it adds the
activity controls one at a time and reports the betweenness coefficient at each
step, then shows the full model WITH and WITHOUT out_degree.


  - If betweenness is already negative once comment_count / tenure enter, and
    stays clearly negative in "FULL controls EXCEPT out_degree", the suppression
    effect is REAL 
  - If it only turns negative once out_degree is added, the sign reversal is
    largely a collinearity artifact 
"""

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

ANALYSIS_FOLDER = Path("ANALYSIS")           # edit if your path differs
SUBREDDITS = ["airfryer", "instantpot", "Roborock", "RobotVacuums", "roomba"]
REFERENCE = "instantpot"
CONTROLS = ["comment_count", "post_count", "out_degree", "tenure_days"]
TEXTUAL = ["avg_sentiment", "avg_text_length", "topical_specialisation"]
COV = "HC3"


# ── load & prep ───────────────────────────────────────────────────────────────
def load():
    frames = []
    for sub in SUBREDDITS:
        d = pd.read_csv(ANALYSIS_FOLDER / f"{sub}_analysis.csv")
        d["subreddit"] = sub
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if "topical_specialisation" not in df.columns and "topical_specialization" in df.columns:
        df = df.rename(columns={"topical_specialization": "topical_specialisation"})
    for c in ["betweenness", "unique_repliers", "mean_score"] + CONTROLS + TEXTUAL:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def prep(df):
    df["log_unique_repliers"] = np.log(df["unique_repliers"] + 1)
    df["ihs_mean_score"] = np.arcsinh(df["mean_score"])
    df["betweenness_z"] = (df["betweenness"] - df["betweenness"].mean()) / df["betweenness"].std()
    dums = pd.get_dummies(df["subreddit"]).astype(float)
    dum_cols = [c for c in dums.columns if c != REFERENCE]
    df = pd.concat([df, dums[dum_cols]], axis=1)
    # build interaction terms (M2 base)
    for d in dum_cols:
        df[f"betweenness_z_x_{d}"] = df["betweenness_z"] * df[d]
    ix = [f"betweenness_z_x_{d}" for d in dum_cols]
    return df, dum_cols, ix


def fit(y, X):
    X = sm.add_constant(X.astype(float))
    return sm.OLS(y.astype(float), X).fit(cov_type=COV)


def vif_of(X, col):
    Xv = sm.add_constant(X.astype(float))
    i = list(Xv.columns).index(col)
    return variance_inflation_factor(Xv.values, i)


# ── robustness ladder ─────────────────────────────────────────────────────────
df = load()
df, DUM, IX = prep(df)
BASE = ["betweenness_z"] + DUM + IX     # M2 base: centrality + dummies + interactions

LADDER = [
    ("M2 base (no controls)",                       []),
    ("+ comment_count",                             ["comment_count"]),
    ("+ comment_count + post_count",                ["comment_count", "post_count"]),
    ("+ comment_count + post_count + tenure",       ["comment_count", "post_count", "tenure_days"]),
    ("FULL controls EXCEPT out_degree",             ["comment_count", "post_count", "tenure_days"] + TEXTUAL),
    ("FULL controls INCLUDING out_degree (thesis M3)", CONTROLS + TEXTUAL),
]

for dv in ["log_unique_repliers", "ihs_mean_score"]:
    print("\n" + "=" * 84)
    print(f"BETWEENNESS ROBUSTNESS  —  DV = {dv}")
    print("=" * 84)
    print(f"  {'specification':<48} {'coef':>9} {'p':>10} {'VIF':>7}")
    print("  " + "-" * 78)
    for tag, extra in LADDER:
        cols = BASE + extra
        mask = df[dv].notna() & df[cols].notna().all(axis=1)
        res = fit(df.loc[mask, dv], df.loc[mask, cols])
        b = res.params["betweenness_z"]
        p = res.pvalues["betweenness_z"]
        v = vif_of(df.loc[mask, cols], "betweenness_z")
        flag = "  <-- sign flips here" if b < 0 and tag != LADDER[0][0] else ""
        print(f"  {tag:<48} {b:>+9.4f} {p:>10.1e} {v:>7.1f}{flag}")

print("\nVerdict:")
print("  Negative already WITHOUT out_degree  -> suppression is real, keep the claim.")
print("  Negative ONLY WITH out_degree present -> collinearity artifact, soften the claim.")

# Save results to CSV for table generation in script 14
rows = []
for dv in ["log_unique_repliers", "ihs_mean_score"]:
    for tag, extra in LADDER:
        cols = BASE + extra
        mask = df[dv].notna() & df[cols].notna().all(axis=1)
        res  = fit(df.loc[mask, dv], df.loc[mask, cols])
        b    = res.params["betweenness_z"]
        p    = res.pvalues["betweenness_z"]
        v    = vif_of(df.loc[mask, cols], "betweenness_z")
        rows.append({
            "dv":    dv,
            "spec":  tag,
            "coef":  round(b, 4),
            "pval":  round(p, 6),
            "vif":   round(v, 1),
        })

pd.DataFrame(rows).to_csv("REGRESSION_RESULTS/betweenness_suppression.csv", index=False)
print("Saved REGRESSION_RESULTS/betweenness_suppression.csv")

print("\n[DONE]")