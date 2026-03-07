import subprocess
import sys

steps = [
    "scripts.generate_prompts",
    "scripts.collect_responses",
    "analysis.stability_analysis",
    "analysis.contradiction_analysis",
    "analysis.final_evaluation",
    "analysis.prompt_sensitivity",
    "analysis.prompt_sensitivity_matrix",
    "analysis.prompt_heatmap",
    "analysis.visualization"
]

for step in steps:
    print(f"\nRunning: {step}\n")

    result = subprocess.run(
        [sys.executable, "-m", step]
    )

    if result.returncode != 0:
        print(f"\nError occurred while running {step}")
        break

print("\nPipeline finished.")