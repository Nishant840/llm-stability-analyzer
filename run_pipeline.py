import subprocess
import sys

steps = [
    "scripts.generate_prompts",
    "scripts.collect_responses",
    "analysis.scripts.stability_analysis",
    "analysis.scripts.contradiction_analysis",
    "analysis.scripts.final_evaluation",
    "analysis.scripts.prompt_sensitivity",
    "analysis.scripts.prompt_sensitivity_matrix",
    "analysis.scripts.prompt_heatmap",
    "analysis.scripts.visualization"
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