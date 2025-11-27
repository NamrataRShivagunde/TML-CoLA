#!/bin/bash

torchrun --standalone --nproc_per_node 1 main_withwandb.py \
    --model_config baseline_configs/llama_350m_aspectratio64.json \
    --model_type llama \
    --lr 0.001 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adamw \
    --run_name baseline-350m \
    --scheduler cosine \
    --offline_mode \
    --offline_data_path datasets/c4/tokenized \
    --tensorboard
