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
    "mixtral-8x7b-32768",
    "groq/compound-mini"
]

TEMPERATURES = [0.1, 0.5, 1.0, 1.5, 2.0]

rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    prompts = list(reader)

for model in MODELS:

    print(f"\nRunning model: {model}\n")

    for temp in TEMPERATURES:
        print(f" Temperature: {temp}")

        for row in tqdm(prompts, leave=False):

            prompt = row["prompt_text"]

            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp
                )

                answer = completion.choices[0].message.content
                length = len(answer.split()) if answer else 0

            except Exception as e:
                print("Error:", e)
                answer = "ERROR"
                length = 0

            rows.append({
                "model": model,
                "temperature": temp,
                "response_length": length,
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
            "temperature",
            "response_length",
            "qid",
            "prompt_type",
            "prompt_text",
            "response"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Responses collected.")