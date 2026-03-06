import pandas as pd

SIM_FILE = "analysis/final_results.csv"
CONTR_FILE = "analysis/contradiction_results.csv"

# load data
sim_df = pd.read_csv(SIM_FILE)
contr_df = pd.read_csv(CONTR_FILE)

# merge datasets
df = sim_df.merge(contr_df, on="qid")

# improved stability score
df["final_stability_score"] = (
    0.6 * df["avg_similarity"]
    + 0.2 * df["min_similarity"]
    - 0.15 * df["std_similarity"]
    - 0.05 * df["contradiction_rate"]
)

# rank questions by stability
df = df.sort_values("final_stability_score", ascending=False)

# save results
df.to_csv("analysis/final_evaluation.csv", index=False)

print("\nFinal Evaluation Results\n")
print(df)