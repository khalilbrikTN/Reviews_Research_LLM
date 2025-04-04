import pandas as pd
from pairwise_f1_score import compute_f1_scores
from nmi_score import compute_nmi_scores
from ari_score import compute_ari_scores

# Load your Excel file
df = pd.read_excel(r"C:\Users\Mohamed Khalil\Desktop\Reviews_Clustering\Reviews_Research_LLM\Data\output_phase1.xlsx")  # Replace with actual path


gpt_col_name = 'Clusters'  # or whatever your column name is
human_col_name = 'Human_Clustering_2'


f1_scores, avg_f1, skipped = compute_f1_scores(df, gpt_col=gpt_col_name, human_col=human_col_name)
nmi_scores, avg_nmi, skipped = compute_nmi_scores(df, gpt_col=gpt_col_name, human_col=human_col_name)
ari_scores, avg_ari, skipped = compute_ari_scores(df, gpt_col=gpt_col_name, human_col=human_col_name)



print('______Report______')

print(f"Average NMI: {avg_nmi:.3f}")
print(f"Average F1-score: {avg_f1:.3f}")
print(f"Average Adjusted Rand Index (ARI): {avg_ari:.3f}")
print(f"Total skipped rows due to errors: {skipped}")
