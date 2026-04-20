import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "analysis/results/final_evaluation.csv"

df = pd.read_csv(INPUT_FILE)

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
    x="qid",
    y="final_stability_score",
    hue="model",
    data=df
)

plt.title("LLM Stability Score by Question")
plt.xlabel("Question ID")
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