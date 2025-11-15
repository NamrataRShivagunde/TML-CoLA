#!/bin/bash
# Hyperparameter tuning script for GPU 1 - CoLA Model
# This script tunes: Stable Steps and Weight Decay

# Base configuration
MODEL_CONFIG="CoLA/cola_configs/cola_60m.json"
MODEL_TYPE="cola"
BATCH_SIZE=128
TOTAL_BATCH_SIZE=512
NUM_TRAINING_STEPS=10000
DTYPE="bfloat16"
EVAL_EVERY=1000
SAVE_EVERY=10000
OPTIMIZER="adamw"
SCHEDULER="warm_stable_decay"
SAVE_DIR="./cola_hp_tuning_results"

# Default values (used when not tuning a specific parameter)
DEFAULT_LR=0.006
DEFAULT_WARMUP=2000
DEFAULT_STABLE=6000
DEFAULT_WD=0.01
DEFAULT_CLIP_GRAD=1.0 # init 0.7 scaling

echo "==================================================================================="
echo "Starting CoLA Hyperparameter Tuning on GPU 1: Stable Steps & Weight Decay"
echo "==================================================================================="

# =======================
# TUNING STABLE STEPS
# =======================
echo ""
echo "### Tuning Stable Steps ###"

# Different stable ratios: 50%, 60%, 65%, 70%, 75% of total steps
STABLE_VALUES=(5000 6500 7000 7500)

for stable in "${STABLE_VALUES[@]}"; do
    echo "-------------------------------------------------------------------"
    echo "Running with stable_steps=$stable"
    echo "-------------------------------------------------------------------"
    
    # Calculate decay steps (total - warmup - stable)
    DECAY_STEPS=$((NUM_TRAINING_STEPS - DEFAULT_WARMUP - stable))
    
    RUN_NAME="cola-60m-wsd-init-scalept5-lr${DEFAULT_LR}-warm${DEFAULT_WARMUP}-stable${stable}-decay${DECAY_STEPS}"
    
    CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 main_withwandb.py \
        --model_type $MODEL_TYPE \
        --model_config $MODEL_CONFIG \
        --lr $DEFAULT_LR \
        --optimizer $OPTIMIZER \
        --batch_size $BATCH_SIZE \
        --total_batch_size $TOTAL_BATCH_SIZE \
        --num_training_steps $NUM_TRAINING_STEPS \
        --warmup_steps $DEFAULT_WARMUP \
        --stable_steps $stable \
        --weight_decay $DEFAULT_WD \
        --dtype $DTYPE \
        --eval_every $EVAL_EVERY \
        --save_every $SAVE_EVERY \
        --grad_clipping $DEFAULT_CLIP_GRAD \
        --scheduler $SCHEDULER \
        --run_name $RUN_NAME \
        --save_dir $SAVE_DIR \
        --single_gpu || {
        echo "WARNING: Run $RUN_NAME failed or was interrupted! Continuing to next run..."
        echo "run_name=$RUN_NAME, lr=$DEFAULT_LR, warmup_steps=$DEFAULT_WARMUP, stable_steps=$stable, weight_decay=$DEFAULT_WD, grad_clipping=$DEFAULT_CLIP_GRAD, final_eval_loss=FAILED, final_eval_perplexity=FAILED" >> $SAVE_DIR/hp_results.txt
    }
    
    echo "Completed: $RUN_NAME"
    echo ""
done

# ========================
# TUNING WEIGHT DECAY
# ========================
echo ""
echo "### Tuning Weight Decay ###"

WD_VALUES=(0.0 0.005 0.02 0.05)

for wd in "${WD_VALUES[@]}"; do
    echo "-------------------------------------------------------------------"
    echo "Running with weight_decay=$wd"
    echo "-------------------------------------------------------------------"
    
    # Calculate decay steps
    DECAY_STEPS=$((NUM_TRAINING_STEPS - DEFAULT_WARMUP - DEFAULT_STABLE))
    
    RUN_NAME="cola-60m-wsd-init-scalept5-lr${DEFAULT_LR}-warm${DEFAULT_WARMUP}-stable${DEFAULT_STABLE}-decay${DECAY_STEPS}-wd${wd}"
    
    CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 main_withwandb.py \
        --model_type $MODEL_TYPE \
        --model_config $MODEL_CONFIG \
        --lr $DEFAULT_LR \
        --optimizer $OPTIMIZER \
        --batch_size $BATCH_SIZE \
        --total_batch_size $TOTAL_BATCH_SIZE \
        --num_training_steps $NUM_TRAINING_STEPS \
        --warmup_steps $DEFAULT_WARMUP \
        --stable_steps $DEFAULT_STABLE \
        --weight_decay $wd \
        --dtype $DTYPE \
        --eval_every $EVAL_EVERY \
        --save_every $SAVE_EVERY \
        --grad_clipping $DEFAULT_CLIP_GRAD \
        --scheduler $SCHEDULER \
        --run_name $RUN_NAME \
        --save_dir $SAVE_DIR \
        --single_gpu || {
        echo "WARNING: Run $RUN_NAME failed or was interrupted! Continuing to next run..."
        echo "run_name=$RUN_NAME, lr=$DEFAULT_LR, warmup_steps=$DEFAULT_WARMUP, stable_steps=$DEFAULT_STABLE, weight_decay=$wd, grad_clipping=$DEFAULT_CLIP_GRAD, final_eval_loss=FAILED, final_eval_perplexity=FAILED" >> $SAVE_DIR/hp_results.txt
    }
    
    echo "Completed: $RUN_NAME"
    echo ""
done

echo "==================================================================================="
echo "GPU 1 CoLA Hyperparameter Tuning Complete!"
echo "Results saved to: $SAVE_DIR/hp_results.txt"
echo "==================================================================================="
