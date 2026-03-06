import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

INPUT_FILE = "analysis/prompt_sensitivity_matrix.csv"

df = pd.read_csv(INPUT_FILE)

heatmap_data = df.pivot(
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

plt.savefig("analysis/prompt_sensitivity_heatmap.png")

plt.show()