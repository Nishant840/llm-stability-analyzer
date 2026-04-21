import csv
import os
import time
from openai import OpenAI
from tqdm import tqdm

INPUT_FILE = "data/prompts/prompts.csv"
OUTPUT_FILE = "data/responses/responses.csv"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Multiple models with fixed valid OpenRouter IDs
MODELS = [
    "meta-llama/llama-3.2-1b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-2-27b-it",
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-oss-120b:free"
]

TEMPERATURES = [0.1, 0.5, 1.0, 1.5, 2.0]
FIELDNAMES = ["model", "temperature", "response_length", "qid", "prompt_type", "prompt_text", "response"]

existing_configs = set()

# Load memory cache from file to avoid re-running API calls
if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_configs.add(
                (row["model"], float(row["temperature"]), row["qid"], row["prompt_type"])
            )
else:
    # Initialize header
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    prompts = list(reader)

for model in MODELS:

    print(f"\nRunning model: {model}\n")

    for temp in TEMPERATURES:
        print(f" Temperature: {temp}")

        for row in tqdm(prompts, leave=False):
            
            # Resume Checkpoint Logic: Skip if already verified
            if (model, float(temp), row["qid"], row["prompt_type"]) in existing_configs:
                continue

            prompt = row["prompt_text"]

            max_retries = 5
            base_delay = 5
            
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temp,
                        max_tokens=600  # Cap extreme length to save tokens
                    )

                    answer = completion.choices[0].message.content
                    length = len(answer.split()) if answer else 0
                    break  # Success, exit the retry loop

                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "rate_limit" in error_msg or "rate limit" in error_msg:
                        delay = base_delay * (2 ** attempt)
                        print(f"   [Rate Limit] Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        print(f"   [API Error] {e}")
                        answer = "ERROR"
                        length = 0
                        break
            else:
                print("   [Failed] Exhausted all retries. Marking as ERROR.")
                answer = "ERROR"
                length = 0

            # Stream instantly to CSV instead of waiting for the end
            with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow({
                    "model": model,
                    "temperature": temp,
                    "response_length": length,
                    "qid": row["qid"],
                    "prompt_type": row["prompt_type"],
                    "prompt_text": prompt,
                    "response": answer
                })
            
            existing_configs.add((model, float(temp), row["qid"], row["prompt_type"]))
            
            # Wait 4 seconds to pace OpenRouter correctly
            time.sleep(4)

print("Responses collected.")