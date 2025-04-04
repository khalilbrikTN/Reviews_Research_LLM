import ast
from itertools import combinations
from sklearn.metrics import f1_score
import pandas as pd

def get_same_cluster_pairs(cluster_dict):
    """Extract all pairs of comments that appear in the same cluster."""
    pairs = set()
    for group in cluster_dict.values():
        comments = [c.strip().lower() for c in group]
        for a, b in combinations(sorted(comments), 2):
            pairs.add(tuple(sorted([a, b])))
    return pairs

def calculate_pairwise_f1(gpt_clusters_str, human_clusters_str):
    """Compute F1 score between GPT and human clustering."""
    gpt_clusters = ast.literal_eval(gpt_clusters_str)
    human_clusters = ast.literal_eval(human_clusters_str)

    gpt_pairs = get_same_cluster_pairs(gpt_clusters)
    human_pairs = get_same_cluster_pairs(human_clusters)

    all_pairs = gpt_pairs.union(human_pairs)

    y_true = [1 if pair in human_pairs else 0 for pair in all_pairs]
    y_pred = [1 if pair in gpt_pairs else 0 for pair in all_pairs]

    return f1_score(y_true, y_pred)

def compute_f1_scores(dataframe, gpt_col='col_name', human_col='col_name'):
    """Compute F1 scores for all rows and return the list and average."""
    f1_scores = []
    skipped = 0

    for i, row in dataframe.iterrows():
        try:
            f1 = calculate_pairwise_f1(row[gpt_col], row[human_col])
            f1_scores.append(f1)
        except Exception:
            skipped += 1

    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return f1_scores, average_f1, skipped
