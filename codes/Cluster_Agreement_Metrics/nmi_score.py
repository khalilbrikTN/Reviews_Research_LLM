
import ast
from sklearn.metrics import normalized_mutual_info_score
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

def calculate_nmi(gpt_clusters_str, human_clusters_str):
    """Compute Normalized Mutual Information between GPT and human clustering."""
    gpt_clusters = ast.literal_eval(gpt_clusters_str)
    human_clusters = ast.literal_eval(human_clusters_str)

    vec1, vec2 = align_label_vectors(gpt_clusters, human_clusters)
    return normalized_mutual_info_score(vec1, vec2)

def compute_nmi_scores(dataframe, gpt_col='Clusters', human_col='Human_Clustering_2'):
    """Compute NMI scores for all rows and return the list and average."""
    nmi_scores = []
    skipped = 0

    for _, row in dataframe.iterrows():
        try:
            nmi = calculate_nmi(row[gpt_col], row[human_col])
            nmi_scores.append(nmi)
        except Exception:
            skipped += 1

    average_nmi = sum(nmi_scores) / len(nmi_scores) if nmi_scores else 0.0
    return nmi_scores, average_nmi, skipped
