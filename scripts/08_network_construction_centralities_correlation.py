import pandas as pd
import networkx as nx
from pathlib import Path
from scipy import stats

# Builds a directed, unweighted reply network for each subreddit and computes three centrality measures for every user in it

# Network definition:
#   - Nodes = users (authors)
#   - Edges = reply relationships (A → B means A replied to B)
#   - Binary edges: whether a reply relationship exists, not how many times
#   - Self-replies excluded
#   - Post authors are only included as nodes if they also appear as comment authors + comments replying to a post (t3_ parent) are included only if the post author also commented elsewhere in the subreddit
#
# Centrality measures:
#   - Betweenness: how often a user lies on the shortest path between others —> captures brokerage positions that bridge otherwise disconnected groups
#   - Eigenvector: how well-connected a user's connections are —> captures embeddedness among structurally prominent others
#   - PageRank: directional version of eigenvector —> weights a user's importance by the prominence of the users who reply to them (damping factor d=0.85)
#
# After computing centrality, Spearman correlations between measures are reported.
# Centrality measures are estimated in separate regression models (see script 13)

DATA_FOLDER = Path("DATA Subreddits")
OUTPUT_FOLDER = Path("NETWORKS")
OUTPUT_FOLDER.mkdir(exist_ok=True)

PAGERANK_ALPHA = 0.85  # standard damping factor from Brin & Page (1998)
MIN_NODES  = 10    # skip networks too small to compute centrality meaningfully


def load_csv(filepath):
    return pd.read_csv(filepath, dtype=str, engine="python", on_bad_lines="skip")


def build_network(df):
    """
    Constructs a directed graph from comment reply relationships.
    Edge direction follows communication flow: replier → parent author.
    Uses a vectorised merge rather than row-by-row iteration for speed.
    """
    G = nx.DiGraph()

    # Map each comment ID to its author
    id_to_author = df[["id", "author"]].drop_duplicates("id").set_index("id")["author"]

    # For each comment, look up who wrote the parent comment
    edges = df[["author", "parent_id_clean"]].copy()
    edges = edges.rename(columns={"author": "replier"})
    edges["parent_author"] = edges["parent_id_clean"].map(id_to_author)

    # Drop rows where parent author is unknown or where user replied to themselves
    edges = edges.dropna(subset=["parent_author"])
    edges = edges[edges["replier"] != edges["parent_author"]]

    # Add edges —> binary, so duplicates are fine (networkx ignores them)
    for replier, parent_author in zip(edges["replier"], edges["parent_author"]):
        G.add_edge(replier, parent_author)

    return G


def compute_centrality(G, name):
    print(f"  Computing centrality: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges ...")

    betweenness = nx.betweenness_centrality(G, normalized=True)

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        # Falls back to in-degree centrality if eigenvector fails to converge,
        # which can happen in very sparse or disconnected networks
        print(f"  [WARNING] Eigenvector did not converge for {name} — using in-degree as fallback")
        eigenvector = nx.in_degree_centrality(G)

    pagerank = nx.pagerank(G, alpha=PAGERANK_ALPHA)

    df_cent = pd.DataFrame({
        "author": list(betweenness.keys()),
        "betweenness": list(betweenness.values()),
        "eigenvector": [eigenvector.get(n, 0) for n in betweenness],
        "pagerank": [pagerank.get(n, 0) for n in betweenness],
        "in_degree": [G.in_degree(n) for n in betweenness],
        "out_degree": [G.out_degree(n) for n in betweenness],
        "subreddit": name,
    }).sort_values("pagerank", ascending=False)

    return df_cent


def network_summary(G, name):
    """Returns basic network statistics as a dict."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    wcc_pct = len(largest_wcc) / n_nodes * 100 if n_nodes > 0 else 0
    avg_in_degree = sum(d for _, d in G.in_degree()) / n_nodes if n_nodes > 0 else 0

    return {
        "subreddit": name,
        "nodes": n_nodes,
        "edges": n_edges,
        "density": round(density, 6),
        "avg_in_degree": round(avg_in_degree, 3),
        "largest_wcc_nodes": len(largest_wcc),
        "largest_wcc_pct": round(wcc_pct, 1),
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

comment_files = sorted(DATA_FOLDER.glob("*_comments_final.csv"))
print(f"Found {len(comment_files)} subreddits to process.\n")

all_centrality = []
all_summaries  = []

for f in comment_files:
    name = f.name.replace("_comments_final.csv", "")
    print(f"── {name}")

    df = load_csv(f)

    required = {"id", "author", "parent_id_clean", "parent_type"}
    if not required.issubset(df.columns):
        print(f"  [SKIPPED] Missing columns: {required - set(df.columns)}\n")
        continue

    G = build_network(df)

    if G.number_of_nodes() < MIN_NODES:
        print(f"  [SKIPPED] Too few nodes ({G.number_of_nodes()})\n")
        continue

    summary = network_summary(G, name)
    all_summaries.append(summary)
    print(f"  Nodes: {summary['nodes']:,} | Edges: {summary['edges']:,} | "
          f"Density: {summary['density']} | LCC: {summary['largest_wcc_pct']}%")

    df_cent = compute_centrality(G, name)
    all_centrality.append(df_cent)

    df_cent.to_csv(OUTPUT_FOLDER / f"{name}_centrality.csv", index=False)
    print(f"  Saved {name}_centrality.csv\n")

# ── Save combined outputs ─────────────────────────────────────────────────────

df_all = pd.concat(all_centrality, ignore_index=True)
df_all.to_csv(OUTPUT_FOLDER / "ALL_centrality.csv", index=False)

df_sum = pd.DataFrame(all_summaries)
df_sum.to_csv(OUTPUT_FOLDER / "NETWORK_SUMMARY.csv", index=False)

print("── Network summary ──────────────────────────────────────")
print(df_sum.to_string(index=False))

# ── Correlation check between centrality measures ─────────────────────────────
# Measures are estimated in separate models, so some correlation is expected and acceptable, but this checks whether any two measures are so highly correlated (r ≥ 0.90) that they are effectively measuring the same thing
# Results are informational — final decisions on model specification are made based on VIF in the regression 

for col in ["betweenness", "eigenvector", "pagerank"]:
    df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

pairs = [
    ("eigenvector", "pagerank"),
    ("betweenness", "eigenvector"),
    ("betweenness", "pagerank"),
]

print("\n\n── Spearman correlations — overall (pooled across subreddits) ────")
overall_results = []
for a, b in pairs:
    clean = df_all[[a, b]].dropna()
    r, p  = stats.spearmanr(clean[a], clean[b])
    flag  = " ← high correlation, check VIF carefully" if abs(r) >= 0.9 else ""
    print(f"  {a} vs {b:<15}  r = {r:.4f}  p = {p:.2e}{flag}")
    overall_results.append({"pair": f"{a}_vs_{b}", "subreddit": "ALL",
                             "spearman_r": round(r, 4), "p_value": round(p, 6)})

print("\n── Spearman correlations — per subreddit (eigenvector vs PageRank) ────")
per_sub_results = []
for sub in sorted(df_all["subreddit"].unique()):
    sub_df = df_all[df_all["subreddit"] == sub][["eigenvector", "pagerank"]].dropna()
    if len(sub_df) < 10:
        continue
    r, p  = stats.spearmanr(sub_df["eigenvector"], sub_df["pagerank"])
    note  = "high" if abs(r) >= 0.9 else "moderate" if abs(r) >= 0.7 else "distinct"
    print(f"  {sub:<20}  r = {r:.4f}  p = {p:.2e}  ({note})")
    per_sub_results.append({"pair": "eigenvector_vs_pagerank", "subreddit": sub,
                             "spearman_r": round(r, 4), "p_value": round(p, 6)})

# Save all correlation results to CSV for reference
results_path = Path("RESULTS")
results_path.mkdir(exist_ok=True)
corr_df = pd.DataFrame(overall_results + per_sub_results)
corr_df.to_csv(results_path / "correlation_check.csv", index=False)
print("\n  Saved RESULTS/correlation_check.csv")

print("\n[DONE] Network construction complete. Outputs in NETWORKS/")
print("       Next: run 09_keyword_dictionary.py")