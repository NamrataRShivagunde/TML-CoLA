#!/bin/bash

torchrun --standalone --nproc_per_node 1 main.py \
    --model_config baseline_configs/llama_130m.json \
    --model_type llama \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adamw \
    --run_name baseline-130m \
    --scheduler cosine \
    --offline_mode \
    --offline_data_path datasets/c4/tokenized \
    --tensorboard
