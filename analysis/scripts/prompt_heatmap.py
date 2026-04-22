import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

INPUT_FILE = "analysis/results/prompt_sensitivity_matrix.csv"

df = pd.read_csv(INPUT_FILE)

agg_df = df.groupby(["prompt1", "prompt2"])["similarity"].mean().reset_index()

heatmap_data = agg_df.pivot(
    index="prompt1",
    columns="prompt2",
    values="similarity"
)

plt.figure(figsize=(8,6))

sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="viridis"
)

plt.title("Prompt Sensitivity Heatmap")
plt.tight_layout()

plt.savefig("analysis/plots/prompt_sensitivity_heatmap.png")

plt.show()