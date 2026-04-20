import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="LLM Stability Analyzer", layout="wide")

st.title("LLM Stability & Robustness Analyzer")

st.write(
"""
Interactive dashboard for analyzing **LLM response stability under prompt variations**.
This framework evaluates multiple models using similarity, variance, contradiction,
and worst-case robustness metrics.
"""
)

final_df = pd.read_csv("analysis/results/final_evaluation.csv")
heatmap_df = pd.read_csv("analysis/results/prompt_sensitivity_matrix.csv")

st.header("Model Stability Leaderboard")

leaderboard = (
    final_df.groupby("model")["final_stability_score"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

st.dataframe(leaderboard, use_container_width=True)

st.header("Model Stability Comparison")

fig, ax = plt.subplots(figsize=(10,5))

sns.barplot(
    data=leaderboard,
    x="model",
    y="final_stability_score",
    hue="model",
    legend=False,
    palette="viridis",
    ax=ax
)

ax.set_ylabel("Average Stability Score")
ax.set_xlabel("Model")
ax.set_title("Average Stability Score by Model")

plt.xticks(rotation=25)

st.pyplot(fig, use_container_width=True)

st.header("Question-wise Stability Explorer")

selected_model = st.selectbox(
    "Select Model",
    final_df["model"].unique()
)

model_df = final_df[final_df["model"] == selected_model]

fig2, ax2 = plt.subplots(figsize=(12,5))

sns.barplot(
    data=model_df,
    x="qid",
    y="final_stability_score",
    hue="qid",
    legend=False,
    palette="magma",
    ax=ax2
)

ax2.set_ylabel("Stability Score")
ax2.set_xlabel("Question ID")
ax2.set_title(f"Stability Scores for {selected_model}")

st.pyplot(fig2, use_container_width=True)

st.header("Worst-Case Robustness")

fig3, ax3 = plt.subplots(figsize=(10,5))

sns.barplot(
    data=final_df,
    x="model",
    y="min_similarity",
    hue="model",
    legend=False,
    palette="coolwarm",
    ax=ax3
)

ax3.set_ylabel("Minimum Similarity")
ax3.set_xlabel("Model")
ax3.set_title("Worst-Case Similarity by Model")

plt.xticks(rotation=25)

st.pyplot(fig3, use_container_width=True)

st.header("Prompt Sensitivity Heatmap")

pivot = heatmap_df.pivot(
    index="prompt1",
    columns="prompt2",
    values="similarity"
)

fig4, ax4 = plt.subplots(figsize=(8,6))

sns.heatmap(
    pivot,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    linewidths=0.5,
    ax=ax4
)

ax4.set_title("Prompt Sensitivity Matrix")

st.pyplot(fig4, use_container_width=True)