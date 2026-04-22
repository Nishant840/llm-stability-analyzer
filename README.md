# LLM Stability & Robustness Analyzer

  

> A research-grade framework for systematically evaluating **Large Language Model response stability** under prompt variations, phrasing noise, and temperature shifts — built as a minor project at IIIT Bhopal.

  

---

  

## Table of Contents

  

- [Overview](#overview)

- [Motivation and Research Questions](#motivation-and-research-questions)

- [Key Features](#key-features)

- [System Architecture](#system-architecture)

- [Models Evaluated](#models-evaluated)

- [Temperature Settings](#temperature-settings)

- [Prompt Variations](#prompt-variations)

- [Dataset Design](#dataset-design)

- [Stability Score Formula](#stability-score-formula)

- [Project Structure](#project-structure)

- [Installation](#installation)

- [Configuration and API Setup](#configuration-and-api-setup)

- [Running the Pipeline](#running-the-pipeline)

- [Running the Dashboard](#running-the-dashboard)

- [Key Findings and Paradoxes](#key-findings-and-paradoxes)

- [Future Work](#future-work)

- [Author](#author)

  

---

  

## Overview

  

The **LLM Stability & Robustness Analyzer** is a fully automated evaluation pipeline that measures how consistently Large Language Models respond when the same question is phrased in different ways or asked at different temperature settings.

  

The framework:

  

- Automatically generates multiple prompt variations for each test question

- Routes all inference through [OpenRouter's](https://openrouter.ai) API, giving access to a wide range of open-source models through a single unified interface

- Computes **semantic similarity** between responses using local Sentence Transformer embeddings

- Detects **logical contradictions** using Facebook's `bart-large-mnli` Natural Language Inference model

- Outputs a single weighted **Stability Score** per model/temperature combination

- Visualizes results through automated plots and an interactive Streamlit dashboard

  

---

  

## Motivation and Research Questions

  

Large Language Models frequently produce noticeably different answers to the same underlying question when its phrasing changes — a phenomenon called **prompt sensitivity**. For example:

  

```

Original: "What is photosynthesis?"

Expert: "Explain photosynthesis as a biology expert."

Brief: "Describe photosynthesis briefly."

Step-by-step: "Explain photosynthesis step by step."

```

  

Even though all four prompts are logically equivalent, the outputs produced by LLMs can vary significantly in wording, depth, structure, and even factual accuracy — depending on the model's parameter count and the temperature setting used during inference.

  

This project was designed to rigorously answer the following research questions:

  

1.  **How stable are LLM responses across rephrased but semantically identical prompts?**

2.  **Does increasing model size (parameter count) always increase output consistency?**

3.  **How often do models contradict themselves across prompt variations?**

4.  **At which temperature thresholds does logical coherence break down?**

5.  **Do models remain stable when prompts contain noise or are semantically empty (gibberish)?**

  

To explore these questions, this project builds a **fault-tolerant, modular, and fully automated robustness evaluation pipeline**.

  

---

  

## Key Features

  

### Fault-Tolerant Response Caching

  

All successfully retrieved LLM responses are checkpointed to disk immediately. If a network timeout or API rate limit interrupts the pipeline mid-run, execution resumes exactly where it left off — no tokens are wasted, and no responses need to be re-fetched.

  

### Prompt Variation Generation

  

For each question in the dataset, the framework automatically generates 6 structurally distinct prompt formats: original, step-by-step, brief, expert, detailed, and noisy. This ensures a consistent and reproducible set of prompt variants across all models and temperatures.

  

### Prompt Noise Robustness Testing

  

A dedicated "noisy" prompt variant injects gibberish or removes all semantic anchors from the question. This tests whether models can maintain coherent, consistent outputs in the absence of meaningful context — a practical proxy for real-world adversarial or poorly written user inputs.

  

### Multi-Model Evaluation Across a Parameter Spectrum

  

Responses are collected from four models spanning 1B to 70B parameters, allowing direct comparison of how model scale relates to output stability. The models are accessed via OpenRouter, so no local GPU resources are required.

  

### Semantic Similarity Analysis

  

Sentence Transformer embeddings (run locally) are used to compute pairwise cosine similarity between all response pairs for each question. This provides a continuous, linguistically-grounded measure of how much responses differ across prompt variants.

  

### Variance and Worst-Case Analysis

  

Beyond average similarity, the pipeline measures the **standard deviation** of pairwise similarities (variance) and the **minimum** pairwise similarity (worst-case divergence) for each model/temperature combination. These metrics help identify fragile or unpredictable model behaviors.

  

### Contradiction Detection

  

All response pairs are evaluated by `facebook/bart-large-mnli`, a large NLI model, to detect cases where one response directly contradicts another for the same question. A high contradiction rate suggests the model is not grounded in a consistent internal representation of the facts.

  

### Weighted Stability Score

  

A single scalar stability score aggregates the above metrics using a weighted formula, making it easy to rank and compare model robustness at a glance.

  

### Prompt Sensitivity Matrix

  

A matrix visualization maps which combinations of prompt style, model, and temperature reliably produce semantically similar outputs — allowing targeted identification of high-risk configurations.

  

### Automated Visualizations

  

The pipeline saves the following plots to disk automatically:

  

-  **Heatmaps** of pairwise similarity across prompt pairs

-  **Violin plots** showing the distribution of similarity scores per model

-  **Scatter plots** tracking similarity drift across temperatures

-  **Bar charts** summarizing stability scores across models and temperatures

  

### Interactive Streamlit Dashboard

  

A local web dashboard enables interactive exploration of all results — filter by model, temperature, or question, and inspect similarity distributions and contradiction patterns dynamically.

  

---

  

## System Architecture

  

The pipeline runs as a sequential series of modular stages:

  

```

┌──────────────────────────────────────────────┐

│ Questions Dataset │

└──────────────────────┬───────────────────────┘

│

▼

┌──────────────────────────────────────────────┐

│ Prompt Generation │ ← 6 variants per question

└──────────────────────┬───────────────────────┘

│

▼

┌──────────────────────────────────────────────┐

│ Multi-Model Response Collection │

│ (OpenRouter API + Checkpoint Cache Engine) │ ← 4 models × 5 temperatures

└──────────────────────┬───────────────────────┘

│

▼

┌──────────────────────────────────────────────┐

│ Semantic Embedding Generation │ ← Local Sentence Transformers

└──────────────────────┬───────────────────────┘

│

▼

┌──────────────────────────────────────────────┐

│ Pairwise Similarity Computation │ ← C(6,2) = 15 pairs per question

└──────────────────────┬───────────────────────┘

│

▼

┌──────────────────────────────────────────────┐

│ Variance and Worst-Case Analysis │ ← Std dev + min similarity

└──────────────────────┬───────────────────────┘

│

▼

┌──────────────────────────────────────────────┐

│ Contradiction Detection │ ← bart-large-mnli NLI model

└──────────────────────┬───────────────────────┘

│

▼

┌──────────────────────────────────────────────┐

│ Weighted Stability Score Computation │

└──────────────────────┬───────────────────────┘

│

▼

┌──────────────────────────────────────────────┐

│ Prompt Sensitivity Matrix Generation │

└──────────────────────┬───────────────────────┘

│

▼

┌──────────────────────────────────────────────┐

│ Visualization and Dashboard │

└──────────────────────────────────────────────┘

```

  

---

  

## Models Evaluated

  

Four models were selected to span a structured spectrum of parameter counts, covering both Meta's Llama ecosystem and a third-party (Google) architecture. All models are accessed via OpenRouter, which normalizes the API interface across providers — ensuring a level playing field for comparison.

  

| Model | Parameters | Role |

| :--- | :---: | :--- |

| `meta-llama/llama-3.2-1b-instruct` | 1B | Ultra-compact baseline — tests the absolute minimum of parametric stability |

| `meta-llama/llama-3.1-8b-instruct` | 8B | Small-weight standard — widely used in edge and consumer deployments |

| `google/gemma-2-27b-it` | 27B | Mid-weight competitor — tests a non-Meta architecture at scale |

| `meta-llama/llama-3.3-70b-instruct` | 70B | Heavyweight flagship — ceiling of open-source LLM capability |

  

---

  

## Temperature Settings

  

Temperature controls how much randomness is injected into the model's next-token selection process. Higher temperature means more diverse, less predictable outputs.

  

Every prompt in this project is submitted at **five distinct temperature levels**, creating a full thermal sweep for each model/question/variant combination:

  

| Temperature | Entropy Level | Behavior |

| :---: | :---: | :--- |

| `0.1` | Near-zero | Near-deterministic. Model almost always picks the highest-probability token. Outputs are highly repetitive and rigid. |

| `0.5` | Low | Mostly consistent outputs with minor surface variation. Sweet spot for factual tasks. |

| `1.0` | Baseline | Default conversational setting. Balances coherence and diversity. Used as the human-equivalent baseline. |

| `1.5` | High | Outputs begin to show creative drift, unusual phrasing, and early-stage hallucination patterns. |

| `2.0` | Maximum | Theoretical upper limit on OpenRouter. Used as a stress test to observe total linguistic and logical collapse. |

  

---

  

## Prompt Variations

  

Each of the 13 test questions is reformulated into 6 structurally distinct prompt styles:

  

| Variant | Description |

| :--- | :--- |

| **Original** | The base question as written, with no modifications |

| **Step-by-step** | Instructs the model to answer using a numbered, sequential breakdown |

| **Brief** | Asks for a concise, minimal response |

| **Expert** | Frames the model as a domain expert answering for a professional audience |

| **Detailed** | Requests a thorough, verbose, and comprehensive answer |

| **Noisy** | Injects gibberish tokens or strips semantic anchors to simulate adversarial or low-quality input |

  

---

  

## Dataset Design

  

### Why a Small, Curated Question Set?

  

The goal of this project is **not** to evaluate LLMs on a large benchmark — it is to validate a **multi-dimensional robustness measurement methodology**. A small, high-quality question set makes the experiment tractable while still producing thousands of evaluation data points.

  

### Scale of the Experiment

  

```

13 questions × 6 prompt variants × 4 models × 5 temperatures

= 1,560 total API responses collected

```

  

### Pairwise Comparisons

  

For each (question, model, temperature) triple, all 6 prompt variant responses are compared pairwise:

  

```

C(6, 2) = 15 pairwise comparisons per (question, model, temperature)

```

  

### Total Similarity Computations

  

```

13 questions × 15 pairs × 4 models × 5 temperatures

= 3,900 semantic similarity computations

```

  

All similarity computations are performed locally using Sentence Transformer models — no external API calls are required during the analysis phase.

  

---

  

## Stability Score Formula

  

The final stability score for a given (model, temperature) combination is computed as a weighted combination of four sub-metrics:

  

```

Stability Score =

0.60 × (Mean Pairwise Similarity)

+ 0.20 × (Minimum Pairwise Similarity)

− 0.15 × (Standard Deviation of Pairwise Similarities)

− 0.05 × (Contradiction Rate)

```

  

**Interpretation of each term:**

  

| Term | Weight | Effect | What It Captures |

| :--- | :---: | :---: | :--- |

| Mean Pairwise Similarity | 0.60 | Positive | Primary driver — high average similarity means consistent answers regardless of phrasing |

| Minimum Pairwise Similarity | 0.20 | Positive | Rewards a high floor — penalizes models that are usually consistent but occasionally produce outlier responses |

| Standard Deviation | 0.15 | Negative | Penalizes high variance — large swings in individual pair scores indicate unreliable behavior |

| Contradiction Rate | 0.05 | Negative | Penalizes NLI-detected contradictions — a self-contradicting model is less trustworthy in production |

  

A score approaching **1.0** indicates highly stable, well-grounded outputs that are robust to prompt phrasing changes. A score near **0** (or negative in extreme cases) indicates an unreliable model configuration.

  

---

  

## Project Structure

  

```

llm-stability-analyzer/

│

├── data/

│ ├── questions/ # Raw question dataset (13 curated questions)

│ ├── prompts/ # Generated prompt variants (6 per question)

│ └── responses/ # Cached LLM responses (checkpointed to disk)

│

├── scripts/

│ ├── clean_errors.py # Identifies and removes malformed/incomplete responses

│ ├── generate_prompts.py # Generates the 6 prompt variants for each question

│ └── collect_responses.py # Handles API calls, retries, and response caching

│

├── analysis/

│ ├── stability_analysis.py # Computes pairwise similarity, variance, worst-case

│ ├── contradiction_analysis.py # Runs bart-large-mnli NLI contradiction detection

│ ├── final_evaluation.py # Aggregates all metrics into the stability score

│ ├── prompt_sensitivity.py # Identifies which prompt types are most destabilizing

│ ├── prompt_sensitivity_matrix.py # Builds the full prompt x model sensitivity matrix

│ ├── prompt_heatmap.py # Generates pairwise similarity heatmaps

│ └── visualization.py # Violin plots, scatter plots, bar charts

│

├── dashboard.py # Interactive Streamlit dashboard

├── verify_models.py # Confirms API access to all four target models

├── run_pipeline.py # Master script — runs all stages end-to-end

└── README.md

```

  

---

  

## Installation

  

**Requirements:** Python 3.10 or higher, a stable internet connection for API inference.

  

### 1. Clone the repository

  

```bash

git  clone  https://github.com/your-username/llm-stability-analyzer.git

cd  llm-stability-analyzer

```

  

### 2. Install Python dependencies

  

```bash

pip  install  -r  requirements.txt

pip  install  openai  streamlit

```

  

> **Note:** The first run will also download the Sentence Transformer and `bart-large-mnli` model weights locally. This is a one-time download (~1–2 GB total). Subsequent runs use the cached weights.

  

---

  

## Configuration and API Setup

  

All LLM inference is routed through [OpenRouter](https://openrouter.ai), which provides a unified API gateway to hundreds of open-source models.

  

### 1. Get an API key

  

Create a free account at [openrouter.ai](https://openrouter.ai) and generate an API key from your dashboard.

  

### 2. Set the environment variable

  

**Linux / macOS:**

  

```bash

export  OPENROUTER_API_KEY="your_api_key_here"

```

  

**Windows (PowerShell):**

  

```powershell

$env:OPENROUTER_API_KEY  =  "your_api_key_here"

```

  

**Windows (Command Prompt):**

  

```cmd

set OPENROUTER_API_KEY=your_api_key_here

```

  

For persistence across sessions, add the export line to your `~/.bashrc` or `~/.zshrc` on Linux/macOS, or set it through System Properties > Environment Variables on Windows.

  

### 3. Verify model access

  

Before running the full pipeline, confirm that your API key has access to all four target models:

  

```bash

python  verify_models.py

```

  

This makes a lightweight test call to each model and reports any access or quota issues. Resolve any failures (usually by enabling the relevant model on your OpenRouter dashboard) before proceeding.

  

---

  

## Running the Pipeline

  

Run the entire pipeline from start to finish with a single command:

  

```bash

python  run_pipeline.py

```

  

**On macOS**, use `caffeinate` to prevent the system from sleeping during long runs:

  

```bash

caffeinate  -i  python  run_pipeline.py

```

  

The pipeline executes the following stages in order:

  

| Step | Stage | Description |

| :---: | :--- | :--- |

| 1 | Prompt Generation | Creates 6 variants for each of the 13 questions |

| 2 | Response Collection | Fetches responses from all 4 models at all 5 temperatures, with automatic checkpointing |

| 3 | Embedding Generation | Encodes all responses using Sentence Transformers (runs locally) |

| 4 | Similarity Computation | Calculates all 3,900 pairwise cosine similarities |

| 5 | Contradiction Detection | Runs NLI analysis on all response pairs |

| 6 | Stability Scoring | Aggregates metrics into final weighted scores |

| 7 | Visualization | Saves heatmaps, violin plots, and bar charts to `output/` |

  

> **Expected runtime:** Approximately 2–4 hours for a full run, dominated by API latency. The checkpoint engine means you can safely interrupt and resume at any point.

  

---

  

## Running the Dashboard

  

Once the pipeline has completed and results are saved, launch the interactive dashboard:

  

```bash

streamlit  run  dashboard.py

```

  

This starts a local web server at `http://localhost:8501` with the following interactive features:

  

| Feature | Description |

| :--- | :--- |

| Model Selector | Filter results by model size (1B, 8B, 27B, 70B) |

| Temperature Slider | Explore how output stability shifts across the five thermal settings |

| Question Browser | Inspect individual questions and view their full response pools |

| Similarity Heatmaps | Visualize pairwise similarity across all 6 prompt variants |

| Stability Leaderboard | Rank all model/temperature combinations by their final stability score |

| Contradiction Explorer | Browse detected contradictions with original response pairs side-by-side |

  

---

  

## Key Findings and Paradoxes

  

Initial analysis of the framework's outputs revealed several counterintuitive findings:

  

### The Linguistic Rigidity Paradox

  

Conventional wisdom assumes that larger models produce more reliable and consistent outputs. However, this framework found the **highest textual stability scores in the 1B parameter model** — not the 70B flagship.

  

The explanation is structural: the 1B model has so few parameters that it lacks the capacity to reformulate sentences creatively when the prompt changes. It produces near-identical, robotic outputs regardless of phrasing. The 70B model, by contrast, uses its vast parameter space to genuinely adapt its response style and depth to each prompt variant — which increases output quality and richness, but reduces raw textual similarity scores.

  

This reveals an important distinction: **semantic consistency** (saying the same things) vs. **surface-level textual consistency** (using the same words). Larger models may be semantically more consistent while appearing textually less stable.

  

### Gibberish Destroys Coherence

  

Prompts that lack any semantic anchors — the "noisy" variant — caused dramatic drops in response consistency across all models and temperatures. Without a factual grounding point, models generate responses from very different regions of their probability distributions, leading to high variance and frequent NLI-detected contradictions.

  

This has a practical implication: models deployed in production should include input validation or prompt hygiene steps to prevent malformed or empty queries from triggering inconsistent behavior.

  

### Factual Anchors Override Temperature

  

For questions with strong, well-known factual answers (e.g., historical facts, physical constants), response consistency remained high even at temperature 1.5. The factual anchor effectively constrains the model's output distribution regardless of the thermal setting.

  

This suggests that temperature matters most for **open-ended or subjective questions**, and has relatively little effect on the consistency of well-grounded factual retrieval.

  

---

  

## Future Work

  

This framework establishes a validated methodology for LLM robustness evaluation. Planned extensions include:

  

| Extension | Description |

| :--- | :--- |

| MoE Architecture Testing | Evaluate Mixture-of-Experts models (e.g., Mixtral, Qwen-MoE) to see if sparse expert routing introduces additional stability patterns |

| Synthetic Dataset Scaling | Automatically generate larger, more diverse question sets to test whether findings generalize beyond the 13-question seed corpus |

| Multi-modal Embeddings | Replace Sentence Transformer embeddings with dense multi-modal representations for richer semantic similarity measurement |

| Statistical Confidence Modeling | Attach confidence intervals and significance tests to all stability scores to distinguish meaningful differences from sampling noise |

| Cross-language Evaluation | Test prompt sensitivity in non-English languages, where tokenization differences may amplify or suppress variance |

  

---

  

## Author

  

**Nishant Kumar**

B.Tech, Computer Science

Indian Institute of Information Technology (IIIT) Bhopal

  

*Minor Project completed under the supervision of*

**Dr. Neeta Anna Eapen**

  

---
