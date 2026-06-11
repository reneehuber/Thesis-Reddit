import pandas as pd
import numpy as np
from pathlib import Path

# Generates all LaTeX tables reported in the thesis.


REGRESSION_FOLDER = Path("REGRESSION_RESULTS")
ANALYSIS_FOLDER   = Path("ANALYSIS")
NETWORK_FOLDER    = Path("NETWORKS")
OUTPUT_FOLDER     = Path("LATEX_TABLES")
OUTPUT_FOLDER.mkdir(exist_ok=True)

REFERENCE   = "instantpot"
CENTRALITY  = ["betweenness", "eigenvector", "pagerank"]
DV_LABELS   = {
    "log_unique_repliers": "Log Unique Repliers",
    "log_mean_score":      "Log Mean Score",
}
CAT_MAP = {
    "airfryer":     "Kitchen",
    "instantpot":   "Kitchen",
    "Roborock":     "Robot Vacuum",
    "RobotVacuums": "Robot Vacuum",
    "roomba":       "Robot Vacuum",
}
VAR_LABELS = {
    "betweenness_z":               "Betweenness (z)",
    "eigenvector_z":               "Eigenvector (z)",
    "pagerank_z":                  "PageRank (z)",
    "betweenness_z_x_Roborock":    "Betweenness (z) x Roborock",
    "betweenness_z_x_RobotVacuums":"Betweenness (z) x RobotVacuums",
    "betweenness_z_x_airfryer":    "Betweenness (z) x airfryer",
    "betweenness_z_x_roomba":      "Betweenness (z) x roomba",
    "eigenvector_z_x_Roborock":    "Eigenvector (z) x Roborock",
    "eigenvector_z_x_RobotVacuums":"Eigenvector (z) x RobotVacuums",
    "eigenvector_z_x_airfryer":    "Eigenvector (z) x airfryer",
    "eigenvector_z_x_roomba":      "Eigenvector (z) x roomba",
    "pagerank_z_x_Roborock":       "PageRank (z) x Roborock",
    "pagerank_z_x_RobotVacuums":   "PageRank (z) x RobotVacuums",
    "pagerank_z_x_airfryer":       "PageRank (z) x airfryer",
    "pagerank_z_x_roomba":         "PageRank (z) x roomba",
    "Roborock":                    "r/Roborock",
    "RobotVacuums":                "r/RobotVacuums",
    "airfryer":                    "r/airfryer",
    "roomba":                      "r/roomba",
    "comment_count":               "Comment Count",
    "post_count":                  "Post Count",
    "out_degree":                  "Out-Degree",
    "tenure_days":                 "Tenure (Days)",
    "avg_sentiment":               "Avg. Sentiment",
    "avg_text_length":             "Avg. Text Length",
    "topical_specialisation":      "Topical Specialisation",
}


def save_tex(filename, content):
    path = OUTPUT_FOLDER / filename
    with open(path, "w") as f:
        f.write(content)
    print(f"  [SAVED] {filename}")


def fmt(val, dec=4):
    if pd.isna(val) or str(val) in ["nan", "", "None"]:
        return "---"
    try:
        return f"{float(val):.{dec}f}"
    except:
        return str(val)


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


def vif_fmt(vif):
    try:
        v = float(vif)
        return f"{v:.2f}" + (r"$\dagger$" if v > 10 else "")
    except:
        return "---"


def var_label(v):
    return VAR_LABELS.get(v, v)


def get_var_group(v):
    if v.endswith("_z") and "_x_" not in v: return 0
    if "_x_" in v:                           return 1
    if v in ["Roborock", "RobotVacuums", "airfryer", "roomba"]: return 2
    if v in ["comment_count", "post_count", "out_degree", "tenure_days"]: return 3
    if v in ["avg_sentiment", "avg_text_length", "topical_specialisation"]: return 4
    return 5


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data ...\n")

df = pd.read_csv(REGRESSION_FOLDER / "regression_results.csv")
for col in df.columns:
    if col not in ["dv", "centrality", "m3_base", "M1_sig", "M2_sig", "M3_sig"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df_det = pd.read_csv(REGRESSION_FOLDER / "regression_results_detailed.csv")
for col in ["coef", "se", "pval", "vif", "r2", "aic", "bic"]:
    if col in df_det.columns:
        df_det[col] = pd.to_numeric(df_det[col], errors="coerce")

df_all = pd.read_csv(ANALYSIS_FOLDER / "ALL_analysis.csv")
for col in df_all.columns:
    df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

net = pd.read_csv(NETWORK_FOLDER / "NETWORK_SUMMARY.csv")
print(f"  Regression results : {len(df)} rows")
print(f"  Detailed results   : {len(df_det)} rows")
print(f"  Analysis data      : {len(df_all):,} users\n")

# Compute N from actual data so table footnotes are always accurate
N_LATEX = f"{len(df_all):,}".replace(",", "{,}")  # e.g. "165{,}568" for LaTeX



# ════════════════════════════════════════════════════════════════════════
# TABLE 1 — DESCRIPTIVE STATISTICS
# ════════════════════════════════════════════════════════════════════════

print("Generating Table 1 ...")

var_groups = [
    ("Dependent Variables", [
        ("unique_repliers", "Unique Repliers"),
        ("mean_score",      "Mean Score"),
    ]),
    ("Centrality Measures (unstandardised)", [
        ("betweenness", "Betweenness Centrality"),
        ("eigenvector", "Eigenvector Centrality"),
        ("pagerank",    "PageRank Centrality"),
    ]),
    ("Control Variables", [
        ("comment_count", "Comment Count"),
        ("post_count",    "Post Count"),
        ("out_degree",    "Out-Degree"),
        ("tenure_days",   "Tenure (Days)"),
    ]),
    ("Textual Covariates", [
        ("avg_text_length",        "Avg. Text Length (Words)"),
        ("avg_sentiment",          "Avg. Sentiment"),
        ("topical_specialisation", "Topical Specialisation"),
    ]),
]

lines = [
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{Descriptive Statistics}", r"\label{tab:descriptives}",
    r"\small", r"\begin{tabular}{lrrrrrr}", r"\hline\hline",
    r"\textbf{Variable} & \textbf{N} & \textbf{Mean} & \textbf{SD} & \textbf{Min} & \textbf{Median} & \textbf{Max} \\",
    r"\hline",
]

for group_label, variables in var_groups:
    lines.append(r"\multicolumn{7}{l}{\textit{" + group_label + r"}} \\")
    for var, label in variables:
        if var not in df_all.columns:
            continue
        s   = df_all[var].dropna()
        dec = 6 if var in ["betweenness", "eigenvector", "pagerank"] else 4
        lines.append(
            r"\quad " + label + " & " + f"{len(s):,}" +
            " & " + f"{s.mean():.{dec}f}" + " & " + f"{s.std():.{dec}f}" +
            " & " + f"{s.min():.{dec}f}" + " & " + f"{s.median():.{dec}f}" +
            " & " + f"{s.max():.{dec}f}" + r" \\"
        )

lines += [
    r"\hline\hline",
    r"\multicolumn{7}{l}{\footnotesize " + f"$N = {N_LATEX}$" + r" users across 5 subreddits.} \\",
    r"\end{tabular}", r"\end{table}",
]
save_tex("table1_descriptives.tex", "\n".join(lines))


# ════════════════════════════════════════════════════════════════════════
# TABLE 2 — NETWORK SUMMARY STATISTICS
# ════════════════════════════════════════════════════════════════════════

print("Generating Table 2 ...")

lines = [
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{Network Summary Statistics by Subreddit}", r"\label{tab:network}",
    r"\small", r"\begin{tabular}{llrrrrr}", r"\hline\hline",
    r"\textbf{Subreddit} & \textbf{Category} & \textbf{Nodes} & \textbf{Edges} & \textbf{Density} & \textbf{Avg.\ In-Degree} & \textbf{LCC (\%)} \\",
    r"\hline",
]
for _, row in net.iterrows():
    sub = row["subreddit"]
    lines.append(
        "r/" + sub + " & " + CAT_MAP.get(sub, "") +
        " & " + f"{int(row['nodes']):,}" +
        " & " + f"{int(row['edges']):,}" +
        " & " + f"{float(row['density']):.2e}" +
        " & " + f"{float(row['avg_in_degree']):.3f}" +
        " & " + f"{float(row['largest_wcc_pct']):.1f}" + r"\% \\"
    )
lines += [
    r"\hline\hline",
    r"\multicolumn{7}{l}{\footnotesize LCC = Largest Connected Component. Directed graph density $= E / [N(N-1)]$.} \\",
    r"\end{tabular}", r"\end{table}",
]
save_tex("table2_network_summary.tex", "\n".join(lines))


# ════════════════════════════════════════════════════════════════════════
# TABLE 3 — SPEARMAN CORRELATION MATRIX
# ════════════════════════════════════════════════════════════════════════

print("Generating Table 3 ...")

corr_vars = [
    "unique_repliers", "mean_score",
    "betweenness", "eigenvector", "pagerank",
    "comment_count", "out_degree", "tenure_days",
    "avg_sentiment", "topical_specialisation",
]
full_labels = [
    "1. Unique Repliers", "2. Mean Score",
    "3. Betweenness", "4. Eigenvector", "5. PageRank",
    "6. Comment Count", "7. Out-Degree", "8. Tenure",
    "9. Sentiment", "10. Topical Spec.",
]
short_labels = [f"({i+1})" for i in range(len(corr_vars))]
corr_matrix  = df_all[corr_vars].corr(method="spearman")
n_c = len(corr_vars)

lines = [
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{Spearman Correlation Matrix}", r"\label{tab:correlations}",
    r"\scriptsize",
    r"\begin{tabular}{l" + "r" * n_c + "}",
    r"\hline\hline",
    r"\textbf{Variable} & " + " & ".join(short_labels) + r" \\",
    r"\hline",
]
for i, var in enumerate(corr_vars):
    cells = [full_labels[i]]
    for j, var2 in enumerate(corr_vars):
        if j > i:
            cells.append("")
        elif j == i:
            cells.append("1.00")
        else:
            cells.append(f"{corr_matrix.loc[var, var2]:.2f}")
    lines.append(" & ".join(cells) + r" \\")
lines += [
    r"\hline\hline",
    r"\multicolumn{" + str(n_c+1) + r"}{l}{\footnotesize Spearman rank correlations. " + f"$N = {N_LATEX}$" + r".} \\",
    r"\end{tabular}", r"\end{table}",
]
save_tex("table3_correlations.tex", "\n".join(lines))


# ════════════════════════════════════════════════════════════════════════
# TABLES 4 & 5 — MAIN REGRESSION RESULTS
# ════════════════════════════════════════════════════════════════════════

def make_results_table(dv, caption, label, filename):
    print(f"Generating {filename} ...")
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{" + caption + "}", r"\label{tab:" + label + "}",
        r"\small", r"\begin{tabular}{lccccccccc}", r"\hline\hline",
        r" & \multicolumn{2}{c}{\textbf{Model 1}}"
        r" & \multicolumn{2}{c}{\textbf{Model 2}}"
        r" & \multicolumn{5}{c}{\textbf{Model 3 (Full)}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-10}",
        r"\textbf{Centrality (z)}"
        r" & \textbf{Coef.} & \textbf{Sig.}"
        r" & \textbf{Coef.} & \textbf{Sig.}"
        r" & \textbf{Coef.} & \textbf{SE} & \textbf{Sig.}"
        r" & \textbf{VIF} & \textbf{$R^{2}$} \\",
        r"\hline",
    ]
    for cent in CENTRALITY:
        r = df[(df["dv"] == dv) & (df["centrality"] == cent)]
        if len(r) == 0:
            lines.append(cent.capitalize() + r" & \multicolumn{9}{c}{---} \\")
            continue
        r = r.iloc[0]
        m1_sig  = sig_stars(r.get("M1_pval"))
        m2_sig  = sig_stars(r.get("M2_pval"))
        m3_sig  = sig_stars(r.get("M3_pval"))
        base    = str(r.get("m3_base", ""))
        note    = r"$^{a}$" if "M1" in base else r"$^{b}$"
        lines.append(
            cent.capitalize() + " & " +
            fmt(r.get("M1_coef"), 4) + " & " + m1_sig + " & " +
            fmt(r.get("M2_coef"), 4) + " & " + m2_sig + " & " +
            fmt(r.get("M3_coef"), 4) + " & " + fmt(r.get("M3_se"), 4) + " & " +
            m3_sig + note + " & " + vif_fmt(r.get("M3_vif")) + " & " +
            fmt(r.get("M3_r2"), 4) + r" \\"
        )
    lines += [
        r"\hline\hline",
        r"\multicolumn{10}{l}{\footnotesize OLS regression on log-transformed DV. " + f"$N = {N_LATEX}$" + r". Reference subreddit: r/instantpot.} \\",
        r"\multicolumn{10}{l}{\footnotesize Centrality measures z-score standardised across pooled dataset (1 unit = 1 SD). Subreddit dummies included in all models.} \\",
        r"\multicolumn{10}{l}{\footnotesize Model 1: centrality + dummies. Model 2: + centrality $\times$ subreddit interactions. Model 3: winning base + controls + textual covariates.} \\",
        r"\multicolumn{10}{l}{\footnotesize *** $p<0.001$, ** $p<0.01$, * $p<0.05$, . $p<0.1$. $\dagger$ VIF $>10$: multicollinearity concern.} \\",
        r"\multicolumn{10}{l}{\footnotesize $^{a}$ M3 built on M1 (LRT: interactions not significant). $^{b}$ M3 built on M2 (LRT: interactions significant, $p<0.05$).} \\",
        r"\end{tabular}", r"\end{table}",
    ]
    save_tex(filename, "\n".join(lines))


make_results_table("log_unique_repliers", "OLS Regression Results: Log Unique Repliers",
                   "results_ur", "table4_log_unique_repliers.tex")
make_results_table("log_mean_score",      "OLS Regression Results: Log Mean Score",
                   "results_ms", "table5_log_mean_score.tex")


# ════════════════════════════════════════════════════════════════════════
# TABLE 6 — LRT SUMMARY
# ════════════════════════════════════════════════════════════════════════

print("Generating Table 6 ...")

lines = [
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{Likelihood Ratio Test: Subreddit $\times$ Centrality Interactions}",
    r"\label{tab:lrt}", r"\small", r"\begin{tabular}{llcccc}", r"\hline\hline",
    r"\textbf{Dependent Variable} & \textbf{Centrality} & \textbf{LRT Stat.} & \textbf{df} & \textbf{$p$-value} & \textbf{M3 Base} \\",
    r"\hline",
]
for dv, dv_label in DV_LABELS.items():
    first = True
    for cent in CENTRALITY:
        r     = df[(df["dv"] == dv) & (df["centrality"] == cent)]
        if len(r) == 0: continue
        r     = r.iloc[0]
        dv_l  = dv_label if first else ""
        first = False
        base  = str(r.get("m3_base", "---")).replace("+controls", "").strip()
        lines.append(
            dv_l + " & " + cent.capitalize() + " & " +
            fmt(r.get("lrt_stat"), 2) + " & 4 & " +
            fmt(r.get("lrt_p"), 4) + " & " + base + r" \\"
        )
    lines.append(r"\hline")

lines += [
    r"\hline",
    r"\multicolumn{6}{l}{\footnotesize LRT compares Model 2 (with interactions) against Model 1. df = 4 interaction terms.} \\",
    r"\multicolumn{6}{l}{\footnotesize M3 base = M2 if LRT $p < 0.05$, else M1. Reference: r/instantpot.} \\",
    r"\end{tabular}", r"\end{table}",
]
save_tex("table6_lrt_summary.tex", "\n".join(lines))


# ════════════════════════════════════════════════════════════════════════
# APPENDIX TABLES — full coefficient tables per model specification
# ════════════════════════════════════════════════════════════════════════

def make_appendix_table(model_key, caption, label, filename, note_extra=""):
    print(f"Generating {filename} ...")
    for dv, dv_label in DV_LABELS.items():
        dv_fn = filename.replace(".tex", "_" + dv.replace("log_", "") + ".tex")
        lines = [
            r"\begin{table}[htbp]", r"\centering",
            r"\caption{" + caption + " --- " + dv_label + "}",
            r"\label{tab:" + label + "_" + dv.replace("log_", "") + "}",
            r"\small", r"\begin{tabular}{lccccc}", r"\hline\hline",
            r"\textbf{Variable} & \textbf{Coef.} & \textbf{SE} & \textbf{$p$-value} & \textbf{Sig.} & \textbf{VIF} \\",
        ]
        for cent in CENTRALITY:
            lines.append(r"\hline")
            lines.append(r"\multicolumn{6}{l}{\textit{Centrality: " + cent.capitalize() + r"}} \\")
            lines.append(r"\hline")

            sub = df_det[
                (df_det["dv"] == dv) &
                (df_det["centrality"] == cent) &
                (df_det["model"] == model_key)
            ].copy()

            if len(sub) == 0:
                lines.append(r"\multicolumn{6}{c}{---} \\")
                continue

            sub["sort_key"] = sub["variable"].apply(get_var_group)
            sub = sub.sort_values("sort_key")

            r2_val  = sub["r2"].iloc[0]  if "r2"  in sub.columns else np.nan
            aic_val = sub["aic"].iloc[0] if "aic" in sub.columns else np.nan

            for _, row in sub.iterrows():
                pval = row.get("pval", np.nan)
                lines.append(
                    var_label(row["variable"]) + " & " +
                    fmt(row["coef"], 4) + " & " + fmt(row["se"], 4) + " & " +
                    fmt(pval, 4) + " & " + (sig_stars(pval) if not pd.isna(pval) else "---") + " & " +
                    vif_fmt(row.get("vif")) + r" \\"
                )
            bic_val = sub["bic"].iloc[0] if "bic" in sub.columns else np.nan
            if not pd.isna(r2_val):
                lines.append(
                    r"\multicolumn{2}{l}{\textit{$R^{2}$ = " + fmt(r2_val, 4) +
                    r"}} & \multicolumn{2}{l}{\textit{AIC = " + fmt(aic_val, 2) +
                    r"}} & \multicolumn{2}{l}{\textit{BIC = " + fmt(bic_val, 2) + r"}} \\"
                )
        lines += [
            r"\hline\hline",
            r"\multicolumn{6}{l}{\footnotesize OLS regression. " + f"$N = {N_LATEX}$" + r". Reference subreddit: r/instantpot.} \\",
            r"\multicolumn{6}{l}{\footnotesize *** $p<0.001$, ** $p<0.01$, * $p<0.05$, . $p<0.1$. $\dagger$ VIF $>10$.} \\",
        ]
        if note_extra:
            lines.append(r"\multicolumn{6}{l}{\footnotesize " + note_extra + r"} \\")
        lines += [r"\end{tabular}", r"\end{table}"]
        save_tex(dv_fn, "\n".join(lines))


make_appendix_table("M1", "Full Regression Output: Model 1 (Base)",
                    "appendix_m1", "tableA1_model1_full.tex",
                    note_extra="Model 1: centrality (z) + subreddit dummies.")

make_appendix_table("M2", "Full Regression Output: Model 2 (Interactions)",
                    "appendix_m2", "tableA2_model2_full.tex",
                    note_extra=r"Model 2: centrality (z) + subreddit dummies + centrality $\times$ subreddit interactions.")

make_appendix_table("M3", "Full Regression Output: Model 3 (Full)",
                    "appendix_m3", "tableA3_model3_full.tex",
                    note_extra="Model 3: winning base (M1 or M2 per LRT) + activity controls + textual covariates.")



# ════════════════════════════════════════════════════════════════════════
# APPENDIX F - SUBREDDIT FIXED EFFECTS FROM MODEL 3
# ════════════════════════════════════════════════════════════════════════

print("Generating Appendix F - Subreddit Fixed Effects ...")

DUMMY_VARS = ["Roborock", "RobotVacuums", "airfryer", "roomba"]
DV_DISPLAY = {
    "log_unique_repliers": "Log Unique Repliers",
    "log_mean_score":      "Log Mean Score",
}
CENT_DISPLAY = {
    "betweenness": "Betweenness",
    "eigenvector": "Eigenvector",
    "pagerank":    "Pagerank",
}

fe_lines = []
fe_lines.append("\\begin{table}[htbp]")
fe_lines.append("\\centering")
fe_lines.append("\\caption{Subreddit Fixed Effects from Model 3 (Reference: r/instantpot)}")
fe_lines.append("\\label{tab:fixed_effects}")
fe_lines.append("\\small")
fe_lines.append("\\begin{tabular}{llcccc}")
fe_lines.append("\\hline\\hline")
fe_lines.append("\\textbf{DV} & \\textbf{Centrality} & \\textbf{Subreddit} & \\textbf{Coef.} & \\textbf{$p$-value} & \\textbf{Sig.} \\\\")
fe_lines.append("\\hline")

for dv, dv_label in DV_DISPLAY.items():
    first_dv = True
    for cent, cent_label in CENT_DISPLAY.items():
        sub = df_det[
            (df_det['dv'] == dv) &
            (df_det['centrality'] == cent) &
            (df_det['model'] == 'M3') &
            (df_det['variable'].isin(DUMMY_VARS))
        ].copy()
        if len(sub) == 0:
            continue
        first_cent = True
        for _, row in sub.iterrows():
            dv_cell   = dv_label   if (first_dv and first_cent) else ''
            cent_cell = cent_label if first_cent else ''
            first_dv   = False
            first_cent = False
            pval = row.get('pval', np.nan)
            sig  = sig_stars(pval) if not pd.isna(pval) else '---'
            fe_lines.append(
                dv_cell + ' & ' + cent_cell + ' & r/' +
                str(row['variable']) + ' & ' +
                fmt(row['coef'], 4) + ' & ' +
                fmt(pval, 4) + ' & ' + sig + ' \\\\'
            )
        fe_lines.append('\\hline')

fe_lines.append("\\hline")
fe_lines.append("\\multicolumn{6}{l}{\\footnotesize Reference category: r/instantpot. Coefficients interpreted relative to instantpot.} \\\\")
fe_lines.append("\\multicolumn{6}{l}{\\footnotesize *** $p<0.001$, ** $p<0.01$, * $p<0.05$, . $p<0.1$.} \\\\")
fe_lines.append("\\end{tabular}")
fe_lines.append("\\end{table}")

save_tex("tableF_subreddit_fixed_effects.tex", "\n".join(fe_lines))

# ── Done ──────────────────────────────────────────────────────────────────────

print(f"\n[DONE] All tables saved to {OUTPUT_FOLDER}/")
print("\nOverleaf preamble: \\usepackage{booktabs}")
print("\nMain body — insert in results section:")
for t in ["table1_descriptives", "table2_network_summary", "table3_correlations",
          "table4_log_unique_repliers", "table5_log_mean_score", "table6_lrt_summary"]:
    print(f"  \\input{{{t}}}")
print("\nAppendix:")
for t in ["tableA1_model1_full_unique_repliers", "tableA1_model1_full_mean_score",
          "tableA2_model2_full_unique_repliers", "tableA2_model2_full_mean_score",
          "tableA3_model3_full_unique_repliers", "tableA3_model3_full_mean_score"]:
    print(f"  \\input{{{t}}}")
print("\nAppendix F:")
for t in ["tableF_subreddit_fixed_effects"]:
    print(f"  \\input{{{t}}}")