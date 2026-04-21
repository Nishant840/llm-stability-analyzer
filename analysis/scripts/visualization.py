import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "analysis/results/final_evaluation.csv"

df = pd.read_csv(INPUT_FILE)
q_df = pd.read_csv("data/questions/questions.csv")
df = df.merge(q_df[["qid", "category"]], on="qid")

plt.figure(figsize=(10,6))

sns.barplot(
    x="model",
    y="final_stability_score",
    hue="model",
    data=df,
    palette="viridis",
    legend=False
)

plt.title("LLM Stability Comparison Across Models")
plt.xlabel("Model")
plt.ylabel("Final Stability Score")

plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("analysis/plots/model_stability_comparison.png")

plt.show()

plt.figure(figsize=(12,6))

sns.barplot(
    x="category",
    y="final_stability_score",
    hue="model",
    data=df
)

plt.title("LLM Stability Score by Category")
plt.xlabel("Category")
plt.ylabel("Final Stability Score")

plt.tight_layout()
plt.savefig("analysis/plots/stability_by_question.png")

plt.show()

plt.figure(figsize=(10,6))

sns.barplot(
    x="model",
    y="min_similarity",
    data=df
)

plt.title("Worst-Case Similarity by Model")
plt.xlabel("Model")
plt.ylabel("Minimum Similarity")

plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("analysis/plots/worst_case_similarity.png")

plt.show()

plt.figure(figsize=(8,5))

sns.histplot(
    df["final_stability_score"],
    bins=8,
    kde=True
)

plt.title("Distribution of Stability Scores")
plt.xlabel("Final Stability Score")

plt.tight_layout()
plt.savefig("analysis/plots/stability_distribution.png")

plt.show()

# NEW Plot: Stability Score vs Temperature
plt.figure(figsize=(10,5))

sns.lineplot(
    data=df,
    x="temperature", 
    y="final_stability_score", 
    hue="model", 
    marker="o"
)

plt.title("Stability Score vs Temperature")
plt.xlabel("Temperature")
plt.ylabel("Final Stability Score")

plt.tight_layout()
plt.savefig("analysis/plots/stability_by_temperature.png")
plt.show()

try:
    res_df = pd.read_csv("data/responses/responses.csv")
    contr_df = pd.read_csv("analysis/results/contradiction_results.csv")
    has_res = True
except FileNotFoundError:
    has_res = False

if has_res and "response_length" in res_df.columns:
    plt.figure(figsize=(10,5))
    sns.lineplot(data=contr_df, x="temperature", y="contradiction_rate", hue="model", marker="o")
    plt.title("Contradiction Rate vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Contradiction Rate")
    plt.tight_layout()
    plt.savefig("analysis/plots/contradictions_by_temperature.png")
    plt.show()

    plt.figure(figsize=(10,5))
    sns.lineplot(data=res_df, x="temperature", y="response_length", hue="model", marker="o")
    plt.title("Response Length vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Response Length")
    plt.tight_layout()
    plt.savefig("analysis/plots/length_by_temperature.png")
    plt.show()

    len_df = res_df.groupby(["model", "temperature", "qid"])["response_length"].mean().reset_index()
    merged_len_contr = len_df.merge(contr_df, on=["model", "temperature", "qid"])
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=merged_len_contr, x="response_length", y="contradiction_rate", hue="model", alpha=0.7)
    plt.title("Response Length vs Contradiction Rate")
    plt.xlabel("Average Response Length")
    plt.ylabel("Contradiction Rate")
    plt.tight_layout()
    plt.savefig("analysis/plots/length_vs_contradiction.png")
    plt.show()

plt.figure(figsize=(10,6))
sns.violinplot(data=df, x="model", y="final_stability_score", hue="model", legend=False, palette="muted", inner="quartile")
plt.title("Stability Score Variance Distribution")
plt.xlabel("Model")
plt.ylabel("Final Stability Score")
plt.tight_layout()
plt.savefig("analysis/plots/stability_variance.png")
plt.show()