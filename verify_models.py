import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODELS = [
    "meta-llama/llama-3.2-1b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-2-27b-it",
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-oss-120b:free"
]

print("Verifying 5 Spectrum Models on OpenRouter...\n")

for model in MODELS:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Ping."}],
            max_tokens=5
        )
        print(f"✅ {model} -> SUCCESS")
    except Exception as e:
        if "402" in str(e):
            print(f"❌ {model} -> INSUFFICIENT CREDITS (402)")
        elif "404" in str(e) or "400" in str(e):
            print(f"❌ {model} -> INVALID MODEL ID")
        else:
            print(f"❌ {model} -> FAILED: {e}")

print("\nVerification Complete!")
