import pandas as pd

df = pd.read_excel(r"C:\Users\Mohamed Khalil\Desktop\RASHID\Reviews_Research_LLM\codes\train_set_label.xlsx")  # Replace with actual file or Google Sheets import
print(df.head())

df["all_comments"] = df[["comment1", "comment2"]].astype(str).agg("; ".join, axis=1)

grouped_df = df.groupby("qid")["all_comments"].apply(list).reset_index()

max_comments = grouped_df["all_comments"].apply(len).max()  # Find max number of comments
grouped_df = pd.DataFrame(grouped_df["all_comments"].tolist(), index=grouped_df["qid"]).reset_index()

grouped_df.columns = ["qid"] + [f"comment_{i+1}" for i in range(max_comments)]

grouped_df.to_csv("grouped_comments.csv", index=False)

pd.read_csv("grouped_comments.csv").to_excel("grouped_comments.xlsx", index=False)

print("Grouping complete! Data saved to 'grouped_comments.csv'.")
