import csv

INPUT_FILE = "data/questions/questions.csv"
OUTPUT_FILE = "data/prompts/prompts.csv"

prompt_templates = {
    "original": "{question}",
    "step_by_step": "Explain step by step: {question}",
    "brief": "Answer briefly: {question}",
    "expert": "You sre an expert. Answer the following question: {question}",
    "detailed": "Provide detailed explanation: {question}"
}

rows = []

with open(INPUT_FILE, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        qid = row["qid"]
        question = row["question"]

        for ptype, template in prompt_templates.items():
            prompt = template.format(question=question)

            rows.append({
                "qid": qid,
                "prompt_type": ptype,
                "prompt_text": prompt
            })

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["qid", "prompt_type", "prompt_text"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Prompts generated successfully.")
