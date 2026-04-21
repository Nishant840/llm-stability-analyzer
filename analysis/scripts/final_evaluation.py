import pandas as pd

SIM_FILE = "analysis/results/final_results.csv"
CONTR_FILE = "analysis/results/contradiction_results.csv"

sim_df = pd.read_csv(SIM_FILE)
contr_df = pd.read_csv(CONTR_FILE)

df = sim_df.merge(contr_df, on=["model", "temperature", "qid"])

df["final_stability_score"] = (
    0.6 * df["avg_similarity"]
    + 0.2 * df["min_similarity"]
    - 0.15 * df["std_similarity"]
    - 0.05 * df["contradiction_rate"]
)

df = df.sort_values("final_stability_score", ascending=False)

df.to_csv("analysis/results/final_evaluation.csv", index=False)

print("\nFinal Evaluation Results\n")
print(df)