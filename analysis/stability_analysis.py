import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import itertools

INPUT_FILE = "data/responses/responses.csv"

df = pd.read_csv(INPUT_FILE)

model = SentenceTransformer("all-MiniLM-L6-v2")

results = []

for qid, group in df.groupby("qid"):

    responses = group["response"].tolist()

    embeddings = model.encode(responses)

    similarities = []

    for i, j in itertools.combinations(range(len(embeddings)), 2):

        sim = cosine_similarity(
            [embeddings[i]],
            [embeddings[j]]
        )[0][0]

        similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities)

    results.append({
        "qid": qid,
        "avg_similarity": avg_similarity
    })

result_df = pd.DataFrame(results)

print(result_df)