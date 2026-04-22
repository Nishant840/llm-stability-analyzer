import subprocess
import sys
import os

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
    if step == "analysis.scripts.contradiction_analysis" and os.path.exists("analysis/results/contradiction_results.csv"):
        print(f"\n⏩ Skipping: {step} (Results already generated! This saves 1 hour of CPU time!)")
        continue

    print(f"\nRunning: {step}\n")

    result = subprocess.run(
        [sys.executable, "-m", step]
    )

    if result.returncode != 0:
        print(f"\nError occurred while running {step}")
        break

print("\nPipeline finished.")