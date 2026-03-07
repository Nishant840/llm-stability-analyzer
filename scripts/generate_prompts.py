import csv
import random

random.seed(42)

INPUT_FILE = "data/questions/questions.csv"
OUTPUT_FILE = "data/prompts/prompts.csv"

prompt_templates = {
    "original": "{question}",
    "step_by_step": "Explain step by step: {question}",
    "brief": "Answer briefly: {question}",
    "expert": "You are an expert. Answer the following question: {question}",
    "detailed": "Provide detailed explanation: {question}"
}

def generate_noisy_prompt(question):
    variants = [
        question.lower(),
        question + " please explain",
        "tell me " + question.lower(),
        question.replace("Explain", "Tell"),
        question + " briefly",
        question + " in simple words",
        "can you explain " + question.lower(),
        question + "??"
    ]
    return random.choice(variants)

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

        noisy_prompt = generate_noisy_prompt(question)

        rows.append({
            "qid": qid,
            "prompt_type": "noisy",
            "prompt_text": noisy_prompt
        })

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["qid", "prompt_type", "prompt_text"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Prompts generated successfully.")