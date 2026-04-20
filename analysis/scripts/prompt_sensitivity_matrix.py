import pandas as pd

INPUT_FILE = "analysis/results/prompt_pair_similarities.csv"

df = pd.read_csv(INPUT_FILE)

matrix_df = (
    df.groupby(["prompt1", "prompt2"])["similarity"]
    .mean()
    .reset_index()
)

print("\nAverage Prompt Pair Similarity\n")
print(matrix_df)

matrix_df.to_csv("analysis/results/prompt_sensitivity_matrix.csv", index=False)