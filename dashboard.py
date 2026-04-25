import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="LLM Stability Analyzer", layout="wide")

# Global figure settings — prevents zoomed-in rendering
plt.rcParams.update({
    "figure.dpi": 80,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

st.title("LLM Stability & Robustness Analyzer")
st.write(
    "Interactive dashboard for analyzing **LLM response stability** under prompt "
    "variations. Evaluates multiple models using similarity, variance, contradiction, "
    "and worst-case robustness metrics."
)

# ── Load data ─────────────────────────────────────────────────────────────────
final_df = pd.read_csv("analysis/results/final_evaluation.csv")
heatmap_df = pd.read_csv("analysis/results/prompt_sensitivity_matrix.csv")
q_df = pd.read_csv("data/questions/questions.csv")
final_df = final_df.merge(q_df[["qid", "category"]], on="qid")

try:
    res_df = pd.read_csv("data/responses/responses.csv")
    has_responses = "response_length" in res_df.columns
except FileNotFoundError:
    res_df = None
    has_responses = False

try:
    contr_df = pd.read_csv("analysis/results/contradiction_results.csv")
    has_contr = True
except FileNotFoundError:
    contr_df = None
    has_contr = False

# ── Section 1: Overall Model Stability ───────────────────────────────────────
st.header("1. Overall Model Stability")
overall_df = (
    final_df.groupby("model")["final_stability_score"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
fig1, ax1 = plt.subplots(figsize=(9, 4))
sns.barplot(
    data=overall_df, x="model", y="final_stability_score",
    hue="model", legend=False, palette="viridis", ax=ax1
)
ax1.set_title("Model vs Mean Stability Score (all temperatures)")
ax1.set_ylim(0, 1.0)
ax1.set_xlabel("Model")
ax1.set_ylabel("Stability Score")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
st.pyplot(fig1, use_container_width=False)

# ── Section 2: Stability vs Temperature ──────────────────────────────────────
st.header("2. Stability Score vs Temperature")
fig2, ax2 = plt.subplots(figsize=(9, 4))
sns.lineplot(
    data=final_df, x="temperature", y="final_stability_score",
    hue="model", marker="o", ax=ax2
)
ax2.set_title("Stability Score Across Temperatures")
ax2.set_xlabel("Temperature")
ax2.set_ylabel("Stability Score")
plt.tight_layout()
st.pyplot(fig2, use_container_width=False)

# ── Section 3: Response Length vs Temperature (conditional) ──────────────────
if has_responses:
    st.header("3. Response Length vs Temperature")
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    sns.lineplot(
        data=res_df, x="temperature", y="response_length",
        hue="model", marker="o", ax=ax3
    )
    ax3.set_title("Average Word Count Across Temperatures")
    ax3.set_xlabel("Temperature")
    ax3.set_ylabel("Word Count")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=False)

# ── Section 4: Detailed Temperature Analysis ─────────────────────────────────
st.header("4. Detailed Temperature Analysis")
selected_temp = st.selectbox(
    "Select Temperature",
    sorted(final_df["temperature"].unique())
)
curr_df = final_df[final_df["temperature"] == selected_temp]

# Leaderboard table
st.subheader("Model Stability Leaderboard")
leaderboard = (
    curr_df.groupby("model")["final_stability_score"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
st.dataframe(leaderboard, use_container_width=False)

# Model stability bar
st.subheader("Model Stability Comparison")
fig4, ax4 = plt.subplots(figsize=(9, 4))
sns.barplot(
    data=leaderboard, x="model", y="final_stability_score",
    hue="model", legend=False, palette="viridis", ax=ax4
)
ax4.set_title(f"Avg Stability Score by Model (T={selected_temp})")
ax4.set_ylim(0, 1.0)
ax4.set_xlabel("Model")
ax4.set_ylabel("Stability Score")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
st.pyplot(fig4, use_container_width=False)

# Category-wise explorer
st.subheader("Category-wise Stability Explorer")
selected_model = st.selectbox("Select Model", curr_df["model"].unique())
model_df = curr_df[curr_df["model"] == selected_model]
fig5, ax5 = plt.subplots(figsize=(9, 4))
sns.barplot(
    data=model_df, x="category", y="final_stability_score",
    hue="category", legend=False, palette="magma", ax=ax5
)
ax5.set_title(f"Category Stability — {selected_model.split('/')[-1]} (T={selected_temp})")
ax5.set_xlabel("Question Category")
ax5.set_ylabel("Stability Score")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
st.pyplot(fig5, use_container_width=False)

# Worst-case robustness
st.subheader("Worst-Case Robustness (Min Similarity)")
fig6, ax6 = plt.subplots(figsize=(9, 4))
sns.barplot(
    data=curr_df, x="model", y="min_similarity",
    hue="model", legend=False, palette="coolwarm", ax=ax6
)
ax6.set_title(f"Worst-Case (Min) Pairwise Similarity (T={selected_temp})")
ax6.set_xlabel("Model")
ax6.set_ylabel("Min Similarity")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
st.pyplot(fig6, use_container_width=False)

# Prompt sensitivity heatmap
st.subheader("Prompt Sensitivity Heatmap")
curr_heatmap = heatmap_df[heatmap_df["temperature"] == selected_temp]
pivot = curr_heatmap.pivot(index="prompt1", columns="prompt2", values="similarity")
fig7, ax7 = plt.subplots(figsize=(9, 5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5, ax=ax7)
ax7.set_title(f"Prompt Pair Similarity (T={selected_temp})")
plt.tight_layout()
st.pyplot(fig7, use_container_width=False)

# ── Section 5: Contradiction Analytics ───────────────────────────────────────
st.header("5. Contradiction & Hallucination Analytics")
if has_contr:
    fig8, ax8 = plt.subplots(figsize=(9, 4))
    sns.lineplot(
        data=contr_df, x="temperature", y="contradiction_rate",
        hue="model", marker="o", ax=ax8
    )
    ax8.set_title("Contradiction Rate vs Temperature")
    ax8.set_xlabel("Temperature")
    ax8.set_ylabel("Contradiction Rate")
    plt.tight_layout()
    st.pyplot(fig8, use_container_width=False)

    if has_responses:
        len_df = (
            res_df.groupby(["model", "temperature", "qid"])["response_length"]
            .mean()
            .reset_index()
        )
        merged = len_df.merge(contr_df, on=["model", "temperature", "qid"])
        fig9, ax9 = plt.subplots(figsize=(9, 4))
        sns.scatterplot(
            data=merged, x="response_length", y="contradiction_rate",
            hue="model", alpha=0.7, ax=ax9
        )
        ax9.set_title("Response Length vs Contradiction Rate")
        ax9.set_xlabel("Avg Response Length (words)")
        ax9.set_ylabel("Contradiction Rate")
        plt.tight_layout()
        st.pyplot(fig9, use_container_width=False)
else:
    st.info("Contradiction results not found. Run contradiction_analysis.py first.")

# ── Section 6: Stability Variance Distribution ────────────────────────────────
st.header("6. Stability Score Distribution")
fig10, ax10 = plt.subplots(figsize=(9, 4))
sns.violinplot(
    data=final_df, x="model", y="final_stability_score",
    hue="model", legend=False, palette="muted", inner="quartile", ax=ax10
)
ax10.set_title("Density and Extremes of Stability Scores")
ax10.set_xlabel("Model")
ax10.set_ylabel("Final Stability Score")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
st.pyplot(fig10, use_container_width=False)