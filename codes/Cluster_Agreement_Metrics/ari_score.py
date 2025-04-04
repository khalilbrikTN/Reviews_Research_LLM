import ast
from itertools import combinations
from sklearn.metrics import adjusted_rand_score
import pandas as pd

def get_label_vector(cluster_dict):
    """Map each comment to a unique integer label based on its cluster."""
    comment_to_cluster = {}
    for cluster_id, (label, comments) in enumerate(cluster_dict.items()):
        for comment in comments:
            comment_to_cluster[comment.strip().lower()] = cluster_id
    return comment_to_cluster

def align_label_vectors(dict1, dict2):
    """Align two label vectors by constructing a unified set of comments."""
    labels1 = get_label_vector(dict1)
    labels2 = get_label_vector(dict2)
    all_comments = sorted(set(labels1) | set(labels2))

    vec1 = [labels1.get(comment, -1) for comment in all_comments]
    vec2 = [labels2.get(comment, -1) for comment in all_comments]
    
    return vec1, vec2

def calculate_ari(gpt_clusters_str, human_clusters_str):
    """Compute Adjusted Rand Index between GPT and human clustering."""
    gpt_clusters = ast.literal_eval(gpt_clusters_str)
    human_clusters = ast.literal_eval(human_clusters_str)

    vec1, vec2 = align_label_vectors(gpt_clusters, human_clusters)
    return adjusted_rand_score(vec1, vec2)

def compute_ari_scores(dataframe, gpt_col='Clusters', human_col='Human_Clustering_2'):
    """Compute ARI scores for all rows and return the list and average."""
    ari_scores = []
    skipped = 0

    for _, row in dataframe.iterrows():
        try:
            ari = calculate_ari(row[gpt_col], row[human_col])
            ari_scores.append(ari)
        except Exception:
            skipped += 1

    average_ari = sum(ari_scores) / len(ari_scores) if ari_scores else 0.0
    return ari_scores, average_ari, skipped
