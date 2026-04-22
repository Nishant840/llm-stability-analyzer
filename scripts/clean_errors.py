import csv
import os

INPUT_FILE = "data/responses/responses.csv"
CLEAN_FILE = "data/responses/responses_clean.csv"

if not os.path.exists(INPUT_FILE):
    print("CSV not found.")
    exit()

VALID_MODELS = [
    "meta-llama/llama-3.2-1b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-2-27b-it",
    "meta-llama/llama-3.3-70b-instruct"
]

cleaned_rows = []
error_count = 0
ghost_count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if row["response"] == "ERROR":
            error_count += 1
        elif row["model"] not in VALID_MODELS:
            ghost_count += 1
        else:
            cleaned_rows.append(row)

with open(CLEAN_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(cleaned_rows)

os.replace(CLEAN_FILE, INPUT_FILE)

print(f"Cleanup complete! Purged:")
print(f"- {error_count} 'ERROR' failures")
print(f"- {ghost_count} Ghost Models entirely removed from CSV")
