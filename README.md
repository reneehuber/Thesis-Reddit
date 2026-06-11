# The Structure of Influence: A Social Network Analysis of Reddit Product Communities

Replication code for the MSc thesis submitted to Rotterdam School of Management, Erasmus University (June 2026).

---

## Data

Data are sourced from the **[Arctic Shift Reddit archive](https://arctic-shift.photon-reddit.com/)**, which provides publicly available historical Reddit submissions and comments in JSONL format.

Download the following files and place them in a folder called `DATA Subreddits/`:

| File | Subreddit |
|---|---|
| `r_Airfryer_comments.jsonl` | r/airfryer |
| `r_Airfryer_posts.jsonl` | r/airfryer |
| `r_instantpot_comments.jsonl` | r/instantpot |
| `r_instantpot_posts.jsonl` | r/instantpot |
| `r_Roborock_comments.jsonl` | r/Roborock |
| `r_Roborock_posts.jsonl` | r/Roborock |
| `r_RobotVacuums_comments.jsonl` | r/RobotVacuums |
| `r_RobotVacuums_posts.jsonl` | r/RobotVacuums |
| `r_roomba_comments.jsonl` | r/roomba |
| `r_roomba_posts.jsonl` | r/roomba |

All data were downloaded on **21 February 2026**, covering each community's full history from creation to that date.

---

## Pipeline

Run scripts in numbered order. Each script reads from the previous stage's output — do not skip steps.

| Script | Description |
|---|---|
| `00_load_sample.py` | Loads a small sample to verify the raw data looks correct before running the full pipeline |
| `01_json_to_csv.py` | Converts JSONL files to CSV — one file per subreddit for comments and posts separately |
| `02_precleaning.py` | Standardizes column formats, ensures ID and text fields are strings, and tags each file with its subreddit name |
| `03_remove_invalid_entries.py` | Removes deleted/removed accounts, bot posts, and rows with missing required fields |
| `04_clean_text.py` | Unicode normalization, whitespace collapsing, and minimum word count filter for comments |
| `05_validate_replies.py` | Strips Reddit's `t1_`/`t3_` prefixes from parent IDs and drops orphaned comments (parent not found) |
| `06_deduplicate.py` | Removes duplicate entries based on unique comment/post ID |
| `07_fix_timestamps.py` | Patches `created_utc` values back from the raw JSONL —> needed for tenure computation + verifies coverage |
| `08_network_construction_centralities_correlation.py` | Builds directed reply networks per subreddit, computes betweenness, eigenvector, and PageRank centrality, and checks intercorrelations between measures |
| `09_keyword_dictionary.py` | Saves product-specific keyword lists to CSV for review before running the variable scripts |
| `10_dependent_variables_and_controls.py` | Computes per-user dependent variables (unique repliers, mean score) and activity controls (comment count, post count, tenure, out-degree) |
| `11_textual_covariates.py` | Computes VADER sentiment, average text length, and topical specialization per user |
| `12_merge.py` | Merges centrality scores, dependent variables, and textual covariates into a single analysis-ready file per subreddit |
| `13_regressions.py` | Runs the full OLS regression pipeline: M1 (base), M2 (interactions), LRT to select M3 base, M3 (full with controls) for both DVs and all three centrality measures |
| `14_tables.py` | Generates all LaTeX tables reported in the thesis (descriptives, network summary, correlation matrix, regression results, LRT table, appendix tables) |

---

## Folder structure

The pipeline creates the following output folders automatically:

```
DATA Subreddits/     raw JSONL and intermediate CSVs
NETWORKS/            centrality scores and network summary statistics
KEYWORDS/            keyword dictionaries 
RESULTS/             per-user dependent variables and textual covariates
ANALYSIS/            merged analysis-ready files (one per subreddit + combined)
REGRESSION_RESULTS/  regression output CSVs and Excel files
LATEX_TABLES/        .tex table files for Overleaf
```

---

## Requirements

```
pandas
numpy
networkx
scipy
statsmodels
vaderSentiment
openpyxl
```

Install with:

```bash
pip install pandas numpy networkx scipy statsmodels vaderSentiment openpyxl
```


---

## Notes

- **Script 07** patches created_utc timestamps back from the original JSONL files. The timestamp field can be lost when loading data with dtype=str during preprocessing. Script 10 depends on these timestamps to compute user tenure, so script 07 must complete successfully before running script 10.
- All regression models use **r/instantpot** as the reference category for subreddit dummies. Centrality measures are z-score standardized across the pooled dataset so that one unit equals one standard deviation.
