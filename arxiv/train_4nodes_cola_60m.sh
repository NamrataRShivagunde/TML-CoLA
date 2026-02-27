#!/usr/bin/env bash

export USER="arrumshi"
export REGION="us-east-1"
export JOB_NAME="arrumshi-cola-debug-v1"
export WORKSPACE="/scratch/scratch/arrumshi/"
#export CONFIG_NAME="small_dense.yaml"
export NUM_NODES="4"

script_parent_path=$(dirname "${BASH_SOURCE[0]}")
source "${script_parent_path}/env_var_setup.sh"

setup_artifacts_dirs ${JOB_NAME} ${WORKSPACE}

# Use absolute path for clarity
REPO_DIR="${WORKSPACE}/TML-CoLA"
cd "${REPO_DIR}"

torchrun --nnodes 4 --nproc_per_node 8 CoLA/main.py \
    --model_type cola \
    --model_config "${REPO_DIR}/CoLA/cola_configs/cola_60m.json" \
    --lr 0.006 \
    --optimizer adamw \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name cola-60m \
    --scheduler cosine \
    --offline_mode \
    --save_dir /checkpoint-fsx/arrumshi \
    --save_every 10000 \
    --tensorboard \
    --offline_data_path /scratch/scratch/arrumshi/datasets/c4/tokenized-35B \
2>&1 | tee "${LOG_DIR}/train.${WORKER_ID}.log"
