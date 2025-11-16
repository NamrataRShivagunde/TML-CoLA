#!/bin/bash
# Hyperparameter tuning script for GPU 1
# This script tunes: Weight Decay and Adam Epsilon

# Base configuration
MODEL_CONFIG="CoLA/baseline_configs/llama_60m.json"
MODEL_TYPE="llama"
BATCH_SIZE=128
TOTAL_BATCH_SIZE=512
NUM_TRAINING_STEPS=10000
DTYPE="bfloat16"
EVAL_EVERY=1000
SAVE_EVERY=20000
OPTIMIZER="adamw"
SCHEDULER="warm_stable_decay"
SAVE_DIR="./hp_tuning_results"

# Default values (used when not tuning a specific parameter)
DEFAULT_LR=0.001
DEFAULT_WARMUP=1000
DEFAULT_STABLE=7000
DEFAULT_WD=0.01

echo "==================================================================================="
echo "Starting Hyperparameter Tuning on GPU 1: Weight Decay & Adam Epsilon"
echo "==================================================================================="

# # =======================
# # TUNING WEIGHT DECAY
# # =======================
# echo ""
# echo "### Tuning Weight Decay ###"

# WD_VALUES=(0.0 0.005 0.05 0.1) # all done

# for wd in "${WD_VALUES[@]}"; do
#     echo "-------------------------------------------------------------------"
#     echo "Running with weight_decay=$wd"
#     echo "-------------------------------------------------------------------"
    
#     # Calculate decay steps (total - warmup - stable)
#     DECAY_STEPS=$((NUM_TRAINING_STEPS - DEFAULT_WARMUP - DEFAULT_STABLE))
    
#     RUN_NAME="baseline-60m-wsd-lr${DEFAULT_LR}-warm${DEFAULT_WARMUP}-decay${DECAY_STEPS}-stable${DEFAULT_STABLE}-wd${wd}"
    
#     CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 CoLA/main_withwandb.py \
#         --model_config $MODEL_CONFIG \
#         --model_type $MODEL_TYPE \
#         --lr $DEFAULT_LR \
#         --batch_size $BATCH_SIZE \
#         --total_batch_size $TOTAL_BATCH_SIZE \
#         --num_training_steps $NUM_TRAINING_STEPS \
#         --warmup_steps $DEFAULT_WARMUP \
#         --stable_steps $DEFAULT_STABLE \
#         --weight_decay $wd \
#         --dtype $DTYPE \
#         --eval_every $EVAL_EVERY \
#         --save_every $SAVE_EVERY \
#         --optimizer $OPTIMIZER \
#         --scheduler $SCHEDULER \
#         --run_name $RUN_NAME \
#         --save_dir $SAVE_DIR \
#         --single_gpu || {
#         echo "WARNING: Run $RUN_NAME failed or was interrupted! Continuing to next run..."
#         echo "run_name=$RUN_NAME, lr=$DEFAULT_LR, warmup_steps=$DEFAULT_WARMUP, stable_steps=$DEFAULT_STABLE, weight_decay=$wd, final_eval_loss=FAILED, final_eval_perplexity=FAILED" >> $SAVE_DIR/hp_results.txt
#     }
    
#     echo "Completed: $RUN_NAME"
#     echo ""
# done

# # ========================
# # TUNING CLIP GRAD
# # ========================
# echo ""
# echo "### Tuning Adam Epsilon ###"

# Note: Adam epsilon may need to be added as an argument to main_withwandb.py
# For now, we'll use beta1 as a proxy or you can add --adam_epsilon argument
CLIP_GRAD_VALUES=(0.1 1.0 5.0)

for clip_grad in "${CLIP_GRAD_VALUES[@]}"; do
    # Calculate decay steps
    DECAY_STEPS=$((NUM_TRAINING_STEPS - DEFAULT_WARMUP - DEFAULT_STABLE))
    
    RUN_NAME="baseline-60m-wsd-lr${DEFAULT_LR}-warm${DEFAULT_WARMUP}-decay${DECAY_STEPS}-stable${DEFAULT_STABLE}-clipgrad${clip_grad}"
    
    # Note: If adam_epsilon is not supported, this will use default
    # You may need to add --adam_epsilon argument to main_withwandb.py
    CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 CoLA/main_withwandb.py \
        --model_config $MODEL_CONFIG \
        --model_type $MODEL_TYPE \
        --lr $DEFAULT_LR \
        --batch_size $BATCH_SIZE \
        --total_batch_size $TOTAL_BATCH_SIZE \
        --num_training_steps $NUM_TRAINING_STEPS \
        --warmup_steps $DEFAULT_WARMUP \
        --stable_steps $DEFAULT_STABLE \
        --weight_decay $DEFAULT_WD \
        --dtype $DTYPE \
        --eval_every $EVAL_EVERY \
        --save_every $SAVE_EVERY \
        --optimizer $OPTIMIZER \
        --scheduler $SCHEDULER \
        --run_name $RUN_NAME \
        --save_dir $SAVE_DIR \
        --grad_clipping $clip_grad \
        --single_gpu || {
        echo "WARNING: Run $RUN_NAME failed or was interrupted! Continuing to next run..."
        echo "run_name=$RUN_NAME, lr=$DEFAULT_LR, warmup_steps=$DEFAULT_WARMUP, stable_steps=$DEFAULT_STABLE, weight_decay=$DEFAULT_WD, final_eval_loss=FAILED, final_eval_perplexity=FAILED" >> $SAVE_DIR/hp_results.txt
    }
    
    echo "Completed: $RUN_NAME"
    echo ""
done

echo "==================================================================================="
echo "GPU 1 Hyperparameter Tuning Complete!"
echo "Results saved to: $SAVE_DIR/hp_results.txt"
echo "==================================================================================="


# # ========================
# # TUNING WARMUP STEPS
# # ========================
# echo ""
# echo "### Tuning Warmup Steps ###"

# WARMUP_VALUES=(500 1000 3000 4000) # done

# for warmup in "${WARMUP_VALUES[@]}"; do
#     echo "-------------------------------------------------------------------"
#     echo "Running with warmup_steps=$warmup"
#     echo "-------------------------------------------------------------------"
    
#     # Adjust stable steps to maintain total steps
#     # stable_steps = total_steps - warmup_steps - decay_buffer
#     # Let's keep decay at 1000 steps minimum
#     STABLE=$((NUM_TRAINING_STEPS - warmup - 1000))
#     DECAY_STEPS=1000
    
#     RUN_NAME="baseline-60m-wsd-lr${DEFAULT_LR}-warm${warmup}-decay${DECAY_STEPS}-stable${STABLE}"
    
#     CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 CoLA/main_withwandb.py \
#         --model_config $MODEL_CONFIG \
#         --model_type $MODEL_TYPE \
#         --lr $DEFAULT_LR \
#         --batch_size $BATCH_SIZE \
#         --total_batch_size $TOTAL_BATCH_SIZE \
#         --num_training_steps $NUM_TRAINING_STEPS \
#         --warmup_steps $warmup \
#         --stable_steps $STABLE \
#         --weight_decay $DEFAULT_WD \
#         --dtype $DTYPE \
#         --eval_every $EVAL_EVERY \
#         --save_every $SAVE_EVERY \
#         --optimizer $OPTIMIZER \
#         --scheduler $SCHEDULER \
#         --run_name $RUN_NAME \
#         --save_dir $SAVE_DIR \
#         --single_gpu || {
#         echo "WARNING: Run $RUN_NAME failed or was interrupted! Continuing to next run..."
#         echo "run_name=$RUN_NAME, lr=$DEFAULT_LR, warmup_steps=$warmup, stable_steps=$STABLE, weight_decay=$DEFAULT_WD, final_eval_loss=FAILED, final_eval_perplexity=FAILED" >> $SAVE_DIR/hp_results.txt
#     }
    
#     echo "Completed: $RUN_NAME"
#     echo ""
# done

# echo "==================================================================================="
# echo "GPU 0 Hyperparameter Tuning Complete!"
# echo "Results saved to: $SAVE_DIR/hp_results.txt"
# echo "==================================================================================="

