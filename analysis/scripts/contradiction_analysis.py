import pandas as pd
import itertools
from transformers import pipeline
from tqdm import tqdm
INPUT_FILE = "data/responses/responses.csv"
OUTPUT_FILE = "analysis/results/contradiction_results.csv"

df = pd.read_csv(INPUT_FILE)

def truncate(text, max_words=200):
    return " ".join(text.split()[:max_words])


print("Loading NLI model...")

nli = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli"
)

results = []
groups = list(df.groupby(["model", "temperature", "qid"]))

for (model, temperature, qid), group in tqdm(groups, desc="Analyzing Contradictions"):

    responses = group["response"].tolist()

    contradictions = 0
    total_pairs = 0

    for r1, r2 in itertools.combinations(responses, 2):

        r1_short = truncate(r1)
        r2_short = truncate(r2)

        result = nli({
            "text": r1_short,
            "text_pair": r2_short
        }, truncation=True)
        # handle both list and dict outputs
        if isinstance(result, list):
            label = result[0]["label"]
        else:
            label = result["label"]

        label = label.lower()

        if "contradiction" in label:
            contradictions += 1

        total_pairs += 1

    contradiction_rate = contradictions / total_pairs

    results.append({
        "model": model,
        "temperature": temperature,
        "qid": qid,
        "contradiction_rate": contradiction_rate
    })

result_df = pd.DataFrame(results)

result_df.to_csv(OUTPUT_FILE, index=False)

print("\nContradiction Results\n")
print(result_df)