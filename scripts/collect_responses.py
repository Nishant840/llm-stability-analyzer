import csv
import os
from groq import Groq
from tqdm import tqdm

INPUT_FILE = "data/prompts/prompts.csv"
OUTPUT_FILE = "data/responses/responses.csv"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Multiple models
MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
    "groq/compound-mini"
]

rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    prompts = list(reader)

for model in MODELS:

    print(f"\nRunning model: {model}\n")

    for row in tqdm(prompts):

        prompt = row["prompt_text"]

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            answer = completion.choices[0].message.content

        except Exception as e:
            print("Error:", e)
            answer = "ERROR"

        rows.append({
            "model": model,                
            "qid": row["qid"],
            "prompt_type": row["prompt_type"],
            "prompt_text": prompt,
            "response": answer
        })

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",                     
            "qid",
            "prompt_type",
            "prompt_text",
            "response"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Responses collected.")