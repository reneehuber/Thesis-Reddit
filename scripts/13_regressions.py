import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

# Runs the full OLS regression pipeline for both dependent variables across all three centrality measures
#
# For each centrality measure and DV, three model specifications are estimated:
#
#   Model 1 (M1): centrality (z-scored) + subreddit dummies
#     — establishes the baseline centrality effect net of community differences
#
#   Model 2 (M2): M1 + centrality × subreddit interaction terms
#     — tests whether the centrality effect varies across communities
#
#   Model 3 (M3): winning base + activity controls + textual covariates
#     — tests whether the centrality effect persists after accounting for participation intensity and content characteristics
#     — base is selected via likelihood ratio test (LRT): if M2 fits significantly better than M1 (p < 0.05), M2 is used as the base for M3; otherwise M1 is used
#
# DV transformations prior to estimation:
#   log_unique_repliers = log(unique_repliers + 1)
#   ihs_mean_score      = arcsinh(mean_score)
#
# Centrality measures are z-score standardised across the pooled dataset, so coefficients reflect the change in the DV associated with a one standard deviation increase in centrality
# -> makes effect sizes comparable across the three measures.
#
# Reference subreddit: r/instantpot (largest community).
# All estimates are relative to this baseline.

ANALYSIS_FOLDER = Path("ANALYSIS")
OUTPUT_FOLDER = Path("REGRESSION_RESULTS")
OUTPUT_FOLDER.mkdir(exist_ok=True)

SUBREDDITS = ["airfryer", "instantpot", "Roborock", "RobotVacuums", "roomba"]
REFERENCE = "instantpot"

CENTRALITY_MEASURES = ["betweenness", "eigenvector", "pagerank"]

# Activity controls — participation intensity
CONTROLS = [
    "comment_count", "post_count", "out_degree", "tenure_days",
]

# Textual covariates — content characteristics
TEXTUAL_COVARIATES = [
    "avg_sentiment", "avg_text_length", "topical_specialisation",
]

# Combined for Model 3
ALL_CONTROLS = CONTROLS + TEXTUAL_COVARIATES

LRT_ALPHA = 0.05  # threshold for selecting M2 as the base for M3


# ── Data loading and preparation ──────────────────────────────────────────────

def load_and_combine():
    frames = []
    for sub in SUBREDDITS:
        df = pd.read_csv(ANALYSIS_FOLDER / f"{sub}_analysis.csv")
        df["subreddit"] = sub
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    for col in CENTRALITY_MEASURES + ["unique_repliers", "mean_score"] + ALL_CONTROLS:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined


def prepare_dvs(df):
    """Transforms both dependent variables prior to regression.
    Unique repliers: log(x + 1) — standard transformation for non-negative counts.
    Mean score: inverse hyperbolic sine (arcsinh) — handles negative values natively without requiring an arbitrary shift constant. 
    """
    df["log_unique_repliers"] = np.log(df["unique_repliers"] + 1)
    df["ihs_mean_score"] = np.arcsinh(df["mean_score"])
    return df


def standardise_centrality(df):
    """Z-scores each centrality measure on the pooled dataset (mean=0, sd=1)."""
    for cent in CENTRALITY_MEASURES:
        mean = df[cent].mean()
        std  = df[cent].std()
        df[f"{cent}_z"] = (df[cent] - mean) / std
    return df


def make_dummies(df):
    """Creates subreddit dummies, dropping the reference category (instantpot)."""
    dummies = pd.get_dummies(df["subreddit"], drop_first=False).astype(float)
    dummy_cols = [c for c in dummies.columns if c != REFERENCE]
    dummies = dummies[dummy_cols]
    return pd.concat([df, dummies], axis=1), dummy_cols


# ── Regression helpers ────────────────────────────────────────────────────────

def compute_vif(X):
    vif = {}
    for i, col in enumerate(X.columns):
        try:
            vif[col] = variance_inflation_factor(X.values, i)
        except Exception:
            vif[col] = np.nan
    return vif


def likelihood_ratio_test(llf_restricted, llf_unrestricted, df_diff):
    """
    Tests whether M2 fits significantly better than M1.
    Under H0 (interactions add nothing), the test statistic is chi-squared
    with degrees of freedom equal to the number of added interaction terms.
    """
    lr_stat = 2 * (llf_unrestricted - llf_restricted)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return round(lr_stat, 4), round(p_value, 6)


def run_ols(y, X, label=""):
    """Fits OLS, returns (result, llf) or (None, None) on failure."""
    X_const = sm.add_constant(X.astype(float))
    try:
        # HC3 heteroskedasticity-robust standard errors — appropriate
        # given the large sample size and transformed DVs
        result = sm.OLS(y, X_const).fit(cov_type="HC3")
        return result, result.llf
    except Exception as e:
        print(f"    [ERROR] {label}: {e}")
        return None, None


def sig_stars(pval):
    try:
        p = float(pval)
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.1:   return "."
        return ""
    except:
        return ""


def extract_centrality_stats(result, cent_z, vif_dict):
    """Pulls coefficient, SE, p-value, and VIF for the centrality variable."""
    if result is None:
        return {"coef": np.nan, "se": np.nan, "pval": np.nan, "sig": "n.c.", "vif": np.nan}
    pval = result.pvalues.get(cent_z, np.nan)
    return {
        "coef": round(result.params.get(cent_z, np.nan), 6),
        "se": round(result.bse.get(cent_z, np.nan), 6),
        "pval": round(pval, 6),
        "sig": sig_stars(pval),
        "vif": round(vif_dict.get(cent_z, np.nan), 4),
    }


def extract_all_coefs(result, vif_dict, dv, centrality, model_label, n_obs, r2, aic, bic):
    """
    Extracts the full coefficient table for every variable in a model.
    Used to build the detailed CSV that feeds into the appendix tables.
    """
    rows = []
    if result is None:
        return rows
    for var in result.params.index:
        if var == "const":
            continue
        pval = result.pvalues.get(var, np.nan)
        rows.append({
            "dv": dv, "centrality": centrality, "model": model_label,
            "n_obs": n_obs, "variable": var,
            "coef": round(float(result.params[var]), 6),
            "se": round(float(result.bse[var]), 6),
            "pval": round(float(pval), 6) if not pd.isna(pval) else np.nan,
            "sig": sig_stars(pval),
            "vif": round(vif_dict.get(var, np.nan), 4),
            "r2": round(r2,  6) if r2  is not None else np.nan,
            "aic": round(aic, 4) if aic is not None else np.nan,
            "bic": round(bic, 4) if bic is not None else np.nan,
        })
    return rows


# ── Load and prepare data ─────────────────────────────────────────────────────

print("=" * 60)
print("Loading and preparing data")
print("=" * 60)

df = load_and_combine()
print(f"  Combined: {len(df):,} users across {df['subreddit'].nunique()} subreddits")

df = prepare_dvs(df)
df = standardise_centrality(df)
df, dummy_cols = make_dummies(df)

print(f"  Subreddit dummies (reference = {REFERENCE}): {dummy_cols}")

for cent in CENTRALITY_MEASURES:
    m = df[cent].mean(); s = df[cent].std()
    print(f"  {cent:<15} mean={m:.6f}  sd={s:.6f}")

# ── Regression loop ───────────────────────────────────────────────────────────

DEPENDENT_VARS = ["log_unique_repliers", "ihs_mean_score"]

all_results = []
all_detail_rows = []

for dv_col in DEPENDENT_VARS:
    print(f"\n{'─'*60}")
    print(f"DV: {dv_col}")
    print(f"{'─'*60}")

    y_full = df[dv_col].copy()

    for cent in CENTRALITY_MEASURES:
        cent_z = f"{cent}_z"
        print(f"\n  Centrality: {cent}")

        all_vars = [cent_z] + dummy_cols + ALL_CONTROLS
        mask = y_full.notna() & df[all_vars].notna().all(axis=1)
        y = y_full[mask]
        df_m = df[mask].copy()
        n = len(y)
        print(f"  N = {n:,}")

        # ── Model 1 ───────────────────────────────────────────────────────────
        X1 = df_m[[cent_z] + dummy_cols]
        vif1 = compute_vif(X1)
        r1, llf1 = run_ols(y, X1, "M1")
        s1 = extract_centrality_stats(r1, cent_z, vif1)
        print(f"  M1  coef={s1['coef']!s:<12} p={s1['pval']!s:<10} {s1['sig']:<4} vif={s1['vif']!s}")

        # ── Model 2 ───────────────────────────────────────────────────────────
        for dum in dummy_cols:
            df_m[f"{cent_z}_x_{dum}"] = df_m[cent_z] * df_m[dum]
        interaction_cols = [f"{cent_z}_x_{dum}" for dum in dummy_cols]

        X2 = df_m[[cent_z] + dummy_cols + interaction_cols]
        vif2 = compute_vif(X2)
        r2_model, llf2 = run_ols(y, X2, "M2")
        s2 = extract_centrality_stats(r2_model, cent_z, vif2)
        print(f"  M2  coef={s2['coef']!s:<12} p={s2['pval']!s:<10} {s2['sig']:<4} vif={s2['vif']!s}")

        # ── Likelihood ratio test ─────────────────────────────────────────────
        lrt_stat, lrt_p = np.nan, np.nan
        use_m2 = False
        if r1 is not None and r2_model is not None and llf1 is not None and llf2 is not None:
            lrt_stat, lrt_p = likelihood_ratio_test(llf1, llf2, len(interaction_cols))
            use_m2 = lrt_p < LRT_ALPHA
            print(f"  LRT stat={lrt_stat}  p={lrt_p}  → M3 base = {'M2' if use_m2 else 'M1'}")
        else:
            print(f"  LRT skipped (convergence failure)")

        # ── Model 3 ───────────────────────────────────────────────────────────
        base_cols = [cent_z] + dummy_cols + (interaction_cols if use_m2 else [])
        base_label = "M2+controls" if use_m2 else "M1+controls"

        X3 = df_m[base_cols + ALL_CONTROLS]
        vif3 = compute_vif(X3)
        r3, llf3 = run_ols(y, X3, "M3")
        s3 = extract_centrality_stats(r3, cent_z, vif3)
        print(f"  M3  coef={s3['coef']!s:<12} p={s3['pval']!s:<10} {s3['sig']:<4} vif={s3['vif']!s}  ({base_label})")

        # ── Store results ─────────────────────────────────────────────────────

        all_detail_rows += extract_all_coefs(r1, vif1, dv_col, cent, "M1", n,
                                             r1.rsquared if r1 is not None else None,
                                             r1.aic if r1 is not None else None,
                                             r1.bic if r1 is not None else None)
        all_detail_rows += extract_all_coefs(r2_model, vif2, dv_col, cent, "M2", n,
                                             r2_model.rsquared if r2_model is not None else None,
                                             r2_model.aic if r2_model is not None else None,
                                             r2_model.bic if r2_model is not None else None)
        all_detail_rows += extract_all_coefs(r3, vif3, dv_col, cent, "M3", n,
                                             r3.rsquared if r3 is not None else None,
                                             r3.aic if r3 is not None else None,
                                             r3.bic if r3 is not None else None)

        # Subreddit fixed effects from M3
        dummy_effects = {}
        if r3 is not None:
            for dum in dummy_cols:
                if dum in r3.params.index:
                    dummy_effects[f"dum_{dum}_coef"] = round(r3.params[dum], 6)
                    dummy_effects[f"dum_{dum}_pval"] = round(r3.pvalues[dum], 6)
                    dummy_effects[f"dum_{dum}_sig"] = sig_stars(r3.pvalues[dum])

        fit = {}
        for lbl, res in [("M1", r1), ("M2", r2_model), ("M3", r3)]:
            if res is not None:
                fit[f"{lbl}_aic"] = round(res.aic, 4)
                fit[f"{lbl}_bic"] = round(res.bic, 4)
                fit[f"{lbl}_r2"] = round(res.rsquared, 6)
                fit[f"{lbl}_llf"] = round(res.llf, 4)  # log-likelihood — used to verify LRT

        row = {
            "dv": dv_col, "centrality": cent, "n_obs": n,
            "reference_sub": REFERENCE,
            "lrt_stat": lrt_stat, "lrt_p": lrt_p, "m3_base": base_label,
            "M1_coef": s1["coef"], "M1_se": s1["se"], "M1_pval": s1["pval"],
            "M1_sig": s1["sig"],  "M1_vif": s1["vif"],
            "M2_coef": s2["coef"], "M2_se": s2["se"], "M2_pval": s2["pval"],
            "M2_sig": s2["sig"],  "M2_vif": s2["vif"],
            "M3_coef": s3["coef"], "M3_se": s3["se"], "M3_pval": s3["pval"],
            "M3_sig": s3["sig"],  "M3_vif": s3["vif"],
        }
        row.update(dummy_effects)
        row.update(fit)
        all_results.append(row)

# ── Save outputs ──────────────────────────────────────────────────────────────

df_results = pd.DataFrame(all_results)
df_results.to_csv(OUTPUT_FOLDER / "regression_results.csv", index=False)
print(f"\n[SAVED] regression_results.csv  ({len(df_results)} model rows)")

df_detail = pd.DataFrame(all_detail_rows)
df_detail.to_csv(OUTPUT_FOLDER / "regression_results_detailed.csv", index=False)
print(f"[SAVED] regression_results_detailed.csv  ({len(df_detail)} variable rows)")

# ── Summary print ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("Model 3 centrality effects — summary")
print(f"{'='*60}")
print(f"{'DV':<22} {'Centrality':<14} {'Coef':>10} {'p':>10} {'Sig':>5} {'VIF':>8}  Base")
print("-" * 80)
for _, r in df_results.iterrows():
    print(f"  {r['dv']:<20} {r['centrality']:<14} "
          f"{str(r.get('M3_coef',''))!s:>10} {str(r.get('M3_pval',''))!s:>10} "
          f"{str(r.get('M3_sig',''))!s:>5} {str(r.get('M3_vif',''))!s:>8}  {r.get('m3_base','')}")

print(f"\n[DONE] All results saved to {OUTPUT_FOLDER}/")
print("       Next: run 14_tables.py")