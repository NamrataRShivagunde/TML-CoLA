#!/bin/bash
# Hyperparameter tuning script for GPU 0 - CoLA Model
# This script tunes: Learning Rate and Warmup Steps

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

# Default values (used when not tuning a specific parameter)
DEFAULT_LR=0.006
DEFAULT_WARMUP=2000
DEFAULT_STABLE=6000
DEFAULT_WD=0.01
DEFAULT_CLIP_GRAD=1.0 # init 0.7 scaling

echo "==================================================================================="
echo "Starting CoLA Hyperparameter Tuning on GPU 0: Learning Rate & Warmup Steps"
echo "==================================================================================="

# ====================
# TUNING LEARNING RATE
# ====================
echo ""
echo "### Tuning Learning Rate ###"

# CoLA typically uses higher LR than baseline due to low-rank structure
LR_VALUES=(0.001 0.01 0.03 0.06 0.1) # default is 0.006

for lr in "${LR_VALUES[@]}"; do
    echo "-------------------------------------------------------------------"
    echo "Running with lr=$lr"
    echo "-------------------------------------------------------------------"
    
    # Calculate decay steps (total - warmup - stable)
    DECAY_STEPS=$((NUM_TRAINING_STEPS - DEFAULT_WARMUP - DEFAULT_STABLE))
    
    RUN_NAME="cola-60m-wsd-init0.7-clipgrad1-lr${lr}-wm${DEFAULT_WARMUP}-st${DEFAULT_STABLE}-dy${DECAY_STEPS}"
    
    CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 CoLA/main_withwandb.py \
        --model_type $MODEL_TYPE \
        --model_config $MODEL_CONFIG \
        --lr $lr \
        --optimizer $OPTIMIZER \
        --batch_size $BATCH_SIZE \
        --total_batch_size $TOTAL_BATCH_SIZE \
        --num_training_steps $NUM_TRAINING_STEPS \
        --warmup_steps $DEFAULT_WARMUP \
        --stable_steps $DEFAULT_STABLE \
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
        echo "run_name=$RUN_NAME, lr=$lr, warmup_steps=$DEFAULT_WARMUP, stable_steps=$DEFAULT_STABLE, weight_decay=$DEFAULT_WD, grad_clipping=$DEFAULT_CLIP_GRAD, final_eval_loss=FAILED, final_eval_perplexity=FAILED" >> $SAVE_DIR/hp_results.txt
    }
    
    echo "Completed: $RUN_NAME"
    echo ""
done

# ========================
# TUNING WARMUP STEPS
# ========================
echo ""
echo "### Tuning Warmup Steps ###"

WARMUP_VALUES=(500 1000 3000 4000)

for warmup in "${WARMUP_VALUES[@]}"; do
    echo "-------------------------------------------------------------------"
    echo "Running with warmup_steps=$warmup"
    echo "-------------------------------------------------------------------"
    
    # Adjust stable steps to maintain total = warmup + stable + decay
    # Keep decay at 1000 steps minimum
    STABLE=$((NUM_TRAINING_STEPS - warmup - 1000))
    DECAY_STEPS=1000
    
    RUN_NAME="cola-60m-wsd-init0.7-clipgrad1-lr${DEFAULT_LR}-wm${warmup}-st${STABLE}-dy${DECAY_STEPS}"
    
    CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 CoLA/main_withwandb.py \
        --model_type $MODEL_TYPE \
        --model_config $MODEL_CONFIG \
        --lr $DEFAULT_LR \
        --optimizer $OPTIMIZER \
        --batch_size $BATCH_SIZE \
        --total_batch_size $TOTAL_BATCH_SIZE \
        --num_training_steps $NUM_TRAINING_STEPS \
        --warmup_steps $warmup \
        --stable_steps $STABLE \
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
        echo "run_name=$RUN_NAME, lr=$DEFAULT_LR, warmup_steps=$warmup, stable_steps=$STABLE, weight_decay=$DEFAULT_WD, grad_clipping=$DEFAULT_CLIP_GRAD, final_eval_loss=FAILED, final_eval_perplexity=FAILED" >> $SAVE_DIR/hp_results.txt
    }
    
    echo "Completed: $RUN_NAME"
    echo ""
done

echo "==================================================================================="
echo "GPU 0 CoLA Hyperparameter Tuning Complete!"
echo "Results saved to: $SAVE_DIR/hp_results.txt"
echo "==================================================================================="
