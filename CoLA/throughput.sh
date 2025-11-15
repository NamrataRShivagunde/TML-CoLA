#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
scripts=(
    "bash scripts/baseline_scripts/baseline350m.sh"
    "bash scripts/baseline_scripts/baseline1b.sh"
    "bash scripts/cola_scripts/cola350m.sh"
    "bash scripts/cola_scripts/cola1b.sh"
)

for s in "${scripts[@]}"; do
    echo "============================================="
    echo "Starting: $s"
    echo "============================================="

    $s
    exit_code=$?

    # If killed from wandb (SIGTERM inside Python), the script will EXIT CODE > 0.
    echo "Exit code = $exit_code"

    # DO NOT stop — continue to next run
    echo "Continuing to next run..."
done
