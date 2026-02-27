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

cd TML-CoLA

torchrun --nnodes 4 --nproc_per_node 8 main.py \
    --model_config baseline_configs/llama_60m.json \
    --model_type llama \
    --lr 0.001 \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adamw \
    --run_name baseline-60m \
    --scheduler cosine \
    --offline_mode \
    --save_dir /checkpoint-fsx/arrumshi \
    --save_every 10000 \
    --tensorboard \
    --offline_data_path /scratch/scratch/arrumshi/datasets/c4/tokenized-35B \
2>&1 | tee "${LOG_DIR}/train.${WORKER_ID}.log"

#    > >(tee "${LOG_DIR}"/train."${WORKER_ID}".log) 2>&1

#torchrun --nnodes=1 --nproc-per-node=8 --rdzv_conf join_timeout=1800,close_timeout=600,timeout=1800 \
# "${WORKSPACE}/AGIModeling-UniversalModel/entrypoints/pretrain_universal.py" \
# --config-path "${WORKSPACE}"/low_rank/conf \
# --config-name ${CONFIG_NAME} \
# > >(tee "${LOG_DIR}"/train."${WORKER_ID}".log) 2>&1

