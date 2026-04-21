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

q_df = pd.read_csv("data/questions/questions.csv")
final_df = final_df.merge(q_df[["qid", "category"]], on="qid")

# Let's see if response tracking exists
try:
    res_df = pd.read_csv("data/responses/responses.csv")
    has_responses = True
except FileNotFoundError:
    has_responses = False

st.header("Stability vs. Temperature")
fig_temp, ax_temp = plt.subplots(figsize=(10, 5))
sns.lineplot(data=final_df, x="temperature", y="final_stability_score", hue="model", marker="o", ax=ax_temp)
ax_temp.set_title("Average Stability Score Across Temperatures")
st.pyplot(fig_temp, use_container_width=True)

if has_responses and "response_length" in res_df.columns:
    st.header("Response Length vs. Temperature")
    fig_len, ax_len = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=res_df, x="temperature", y="response_length", hue="model", marker="o", ax=ax_len)
    ax_len.set_title("Average Token Length Across Temperatures")
    st.pyplot(fig_len, use_container_width=True)

st.header("Detailed Temperature Analysis")
selected_temp = st.selectbox("Select Temperature for Detailed View", sorted(final_df["temperature"].unique()))

curr_df = final_df[final_df["temperature"] == selected_temp]

st.subheader("Model Stability Leaderboard")

leaderboard = (
    curr_df.groupby("model")["final_stability_score"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

st.dataframe(leaderboard, use_container_width=True)

st.subheader("Model Stability Comparison")

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

st.subheader("Category-wise Stability Explorer")

selected_model = st.selectbox(
    "Select Model",
    curr_df["model"].unique()
)

model_df = curr_df[curr_df["model"] == selected_model]

fig2, ax2 = plt.subplots(figsize=(12,5))

sns.barplot(
    data=model_df,
    x="category",
    y="final_stability_score",
    hue="category",
    legend=False,
    palette="magma",
    ax=ax2
)

ax2.set_ylabel("Stability Score")
ax2.set_xlabel("Question Category")
ax2.set_title(f"Stability Scores for {selected_model}")

st.pyplot(fig2, use_container_width=True)

st.subheader("Worst-Case Robustness")

fig3, ax3 = plt.subplots(figsize=(10,5))

sns.barplot(
    data=curr_df,
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

st.subheader("Prompt Sensitivity Heatmap")

curr_heatmap_df = heatmap_df[heatmap_df["temperature"] == selected_temp]

pivot = curr_heatmap_df.pivot(
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

st.header("Contradiction & Hallucination Analytics")

try:
    contr_df = pd.read_csv("analysis/results/contradiction_results.csv")
    has_contr = True
except FileNotFoundError:
    has_contr = False

if has_contr:
    st.subheader("Contradiction Rate vs Temperature")
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=contr_df, x="temperature", y="contradiction_rate", hue="model", marker="o", ax=ax5)
    ax5.set_title("How Temperature Impacts Contradictions")
    st.pyplot(fig5, use_container_width=True)

    if has_responses and "response_length" in res_df.columns:
        st.subheader("Response Length vs Contradiction Rate")
        len_df = res_df.groupby(["model", "temperature", "qid"])["response_length"].mean().reset_index()
        merged_len_contr = len_df.merge(contr_df, on=["model", "temperature", "qid"])
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=merged_len_contr, x="response_length", y="contradiction_rate", hue="model", alpha=0.7, ax=ax6)
        ax6.set_title("Does verbosity lead to contradictions?")
        st.pyplot(fig6, use_container_width=True)

st.subheader("Stability Variance Distribution (Violin Plot)")
fig7, ax7 = plt.subplots(figsize=(10, 6))
sns.violinplot(data=final_df, x="model", y="final_stability_score", hue="model", legend=False, palette="muted", inner="quartile", ax=ax7)
ax7.set_title("Density and Extremes of Stability Scores")
st.pyplot(fig7, use_container_width=True)