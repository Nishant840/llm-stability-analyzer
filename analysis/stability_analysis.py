import pandas as pd
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.embedding_model import get_embeddings

INPUT_FILE = "data/responses/responses.csv"
OUTPUT_FILE = "analysis/final_results.csv"

df = pd.read_csv(INPUT_FILE)

results = []

for qid, group in df.groupby("qid"):

    responses = group["response"].tolist()

    embeddings = get_embeddings(responses)

    similarities = []

    for i, j in itertools.combinations(range(len(embeddings)), 2):

        sim = cosine_similarity(
            [embeddings[i]],
            [embeddings[j]]
        )[0][0]

        similarities.append(sim)

    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    stability_score = avg_similarity - std_similarity

    results.append({
        "qid": qid,
        "avg_similarity": avg_similarity,
        "std_similarity": std_similarity,
        "stability_score": stability_score
    })

result_df = pd.DataFrame(results)

result_df.to_csv(OUTPUT_FILE, index=False)

print(result_df)