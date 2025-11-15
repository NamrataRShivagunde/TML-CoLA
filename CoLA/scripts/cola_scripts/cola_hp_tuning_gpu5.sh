#!/bin/bash
# Hyperparameter tuning script for GPU 3 - CoLA Model
# This script tunes: Combined Configurations (Best combinations from initial sweeps)

# Base configuration
MODEL_CONFIG="CoLA/cola_configs/cola_60m.json"
MODEL_TYPE="cola"
BATCH_SIZE=128
TOTAL_BATCH_SIZE=512
NUM_TRAINING_STEPS=10000
DTYPE="bfloat16"
EVAL_EVERY=1000
SAVE_EVERY=20000
OPTIMIZER="adamw"
SCHEDULER="warm_stable_decay"
SAVE_DIR="./cola_hp_tuning_results"

echo "==================================================================================="
echo "Starting CoLA Hyperparameter Tuning on GPU 3: Combined Configs"
echo "==================================================================================="

# ========================================
# COMBINED CONFIGURATIONS TO TRY
# ========================================
echo ""
echo "### Testing Combined Hyperparameter Configurations ###"

# Config 1: Higher LR + Higher Weight Decay
echo "-------------------------------------------------------------------"
echo "Config 1: High LR + High WD"
echo "-------------------------------------------------------------------"
RUN_NAME="cola-60m-wsd-init-scalept7-lr0.01-warm2000-stable6000-decay2000-wd0.02-clipgrad1"
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 CoLA/main_withwandb.py \
    --model_type $MODEL_TYPE \
    --model_config $MODEL_CONFIG \
    --lr 0.01 \
    --optimizer $OPTIMIZER \
    --batch_size $BATCH_SIZE \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --num_training_steps $NUM_TRAINING_STEPS \
    --warmup_steps 2000 \
    --stable_steps 6000 \
    --weight_decay 0.02 \
    --dtype $DTYPE \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --grad_clipping 1.0 \
    --scheduler $SCHEDULER \
    --run_name $RUN_NAME \
    --save_dir $SAVE_DIR \
    --single_gpu || {
    echo "WARNING: Run $RUN_NAME failed or was interrupted! Continuing to next run..."
    echo "run_name=$RUN_NAME, lr=0.008, warmup_steps=2000, stable_steps=6000, weight_decay=0.02, grad_clipping=0.5, final_eval_loss=FAILED, final_eval_perplexity=FAILED" >> $SAVE_DIR/hp_results.txt
}
echo "Completed: $RUN_NAME"
echo ""

# Config 2: Lower LR + Longer Warmup
echo "-------------------------------------------------------------------"
echo "Config 2: Lower LR + Longer Warmup"
echo "-------------------------------------------------------------------"
RUN_NAME="cola-60m-wsd-init-scalept5-lr0.003-warm3000-stable6000-decay1000-wd0.01-clipgrad0.5"
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 main_withwandb.py \
    --model_type $MODEL_TYPE \
    --model_config $MODEL_CONFIG \
    --lr 0.003 \
    --optimizer $OPTIMIZER \
    --batch_size $BATCH_SIZE \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --num_training_steps $NUM_TRAINING_STEPS \
    --warmup_steps 3000 \
    --stable_steps 6000 \
    --weight_decay 0.01 \
    --dtype $DTYPE \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --grad_clipping 0.5 \
    --scheduler $SCHEDULER \
    --run_name $RUN_NAME \
    --save_dir $SAVE_DIR \
    --single_gpu || {
    echo "WARNING: Run $RUN_NAME failed or was interrupted! Continuing to next run..."
    echo "run_name=$RUN_NAME, lr=0.003, warmup_steps=3000, stable_steps=6000, weight_decay=0.01, grad_clipping=0.5, final_eval_loss=FAILED, final_eval_perplexity=FAILED" >> $SAVE_DIR/hp_results.txt
}
echo "Completed: $RUN_NAME"
echo ""

# Config 3: Medium LR + Longer Stable + Higher Clip
echo "-------------------------------------------------------------------"
echo "Config 3: Medium LR + Longer Stable + Higher Clip"
echo "-------------------------------------------------------------------"
RUN_NAME="cola-60m-wsd-init-scalept5-lr0.006-warm2000-stable7000-decay1000-wd0.01-clipgrad1.0"
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 main_withwandb.py \
    --model_type $MODEL_TYPE \
    --model_config $MODEL_CONFIG \
    --lr 0.006 \
    --optimizer $OPTIMIZER \
    --batch_size $BATCH_SIZE \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --num_training_steps $NUM_TRAINING_STEPS \
    --warmup_steps 2000 \
    --stable_steps 7000 \
    --weight_decay 0.01 \
    --dtype $DTYPE \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --grad_clipping 1.0 \
    --scheduler $SCHEDULER \
    --run_name $RUN_NAME \
    --save_dir $SAVE_DIR \
    --single_gpu || {
    echo "WARNING: Run $RUN_NAME failed or was interrupted! Continuing to next run..."
    echo "run_name=$RUN_NAME, lr=0.006, warmup_steps=2000, stable_steps=7000, weight_decay=0.01, grad_clipping=1.0, final_eval_loss=FAILED, final_eval_perplexity=FAILED" >> $SAVE_DIR/hp_results.txt
}
echo "Completed: $RUN_NAME"
echo ""

# Config 4: High LR + No Weight Decay + Lower Clip
echo "-------------------------------------------------------------------"
echo "Config 4: High LR + No WD + Lower Clip"
echo "-------------------------------------------------------------------"
RUN_NAME="cola-60m-wsd-init-scalept5-lr0.01-warm2000-stable6000-decay2000-wd0.0-clipgrad0.3"
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 main_withwandb.py \
    --model_type $MODEL_TYPE \
    --model_config $MODEL_CONFIG \
    --lr 0.01 \
    --optimizer $OPTIMIZER \
    --batch_size $BATCH_SIZE \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --num_training_steps $NUM_TRAINING_STEPS \
    --warmup_steps 2000 \
    --stable_steps 6000 \
    --weight_decay 0.0 \
    --dtype $DTYPE \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --grad_clipping 0.3 \
    --scheduler $SCHEDULER \
    --run_name $RUN_NAME \
    --save_dir $SAVE_DIR \
    --single_gpu || {
    echo "WARNING: Run $RUN_NAME failed or was interrupted! Continuing to next run..."
    echo "run_name=$RUN_NAME, lr=0.01, warmup_steps=2000, stable_steps=6000, weight_decay=0.0, grad_clipping=0.3, final_eval_loss=FAILED, final_eval_perplexity=FAILED" >> $SAVE_DIR/hp_results.txt
}
echo "Completed: $RUN_NAME"
echo ""

# Config 5: Conservative - Lower everything
echo "-------------------------------------------------------------------"
echo "Config 5: Conservative Settings"
echo "-------------------------------------------------------------------"
RUN_NAME="cola-60m-wsd-init-scalept5-lr0.005-warm1500-stable7500-decay1000-wd0.005-clipgrad0.5"
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 main_withwandb.py \
    --model_type $MODEL_TYPE \
    --model_config $MODEL_CONFIG \
    --lr 0.005 \
    --optimizer $OPTIMIZER \
    --batch_size $BATCH_SIZE \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --num_training_steps $NUM_TRAINING_STEPS \
    --warmup_steps 1500 \
    --stable_steps 7500 \
    --weight_decay 0.005 \
    --dtype $DTYPE \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --grad_clipping 0.5 \
    --scheduler $SCHEDULER \
    --run_name $RUN_NAME \
    --save_dir $SAVE_DIR \
    --single_gpu || {
    echo "WARNING: Run $RUN_NAME failed or was interrupted! Continuing to next run..."
    echo "run_name=$RUN_NAME, lr=0.005, warmup_steps=1500, stable_steps=7500, weight_decay=0.005, grad_clipping=0.5, final_eval_loss=FAILED, final_eval_perplexity=FAILED" >> $SAVE_DIR/hp_results.txt
}
echo "Completed: $RUN_NAME"
echo ""

echo "==================================================================================="
echo "GPU 3 CoLA Hyperparameter Tuning Complete!"
echo "Results saved to: $SAVE_DIR/hp_results.txt"
echo "==================================================================================="
