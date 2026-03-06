import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "analysis/final_evaluation.csv"

df = pd.read_csv(INPUT_FILE)
df = df.sort_values("final_stability_score", ascending=False)

plt.figure(figsize=(10,6))

sns.barplot(
    x="qid",
    y="final_stability_score",
    data=df,
    palette="viridis"
)

plt.title("LLM Stability Score by Question")
plt.xlabel("Question ID")
plt.ylabel("Stability Score")

plt.tight_layout()
plt.savefig("analysis/stability_bar_chart.png")

plt.show()

plt.figure(figsize=(8,5))

sns.histplot(
    df["stability_score"],
    bins=8,
    kde=True
)

plt.title("Distribution of Stability Scores")
plt.xlabel("Stability Score")

plt.tight_layout()
plt.savefig("analysis/stability_distribution.png")

plt.show()