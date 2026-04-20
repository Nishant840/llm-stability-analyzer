import pandas as pd
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.embedding_model import get_embeddings

INPUT_FILE = "data/responses/responses.csv"

df = pd.read_csv(INPUT_FILE)

pairs = []

for qid, group in df.groupby("qid"):

    responses = group["response"].tolist()
    prompt_types = group["prompt_type"].tolist()

    embeddings = get_embeddings(responses)

    for i, j in itertools.combinations(range(len(responses)), 2):

        sim = cosine_similarity(
            [embeddings[i]],
            [embeddings[j]]
        )[0][0]

        pairs.append({
            "qid": qid,
            "prompt1": prompt_types[i],
            "prompt2": prompt_types[j],
            "similarity": sim
        })

pairs_df = pd.DataFrame(pairs)

print("\nPrompt Pair Similarities\n")
print(pairs_df.head())

pairs_df.to_csv("analysis/results/prompt_pair_similarities.csv", index=False)