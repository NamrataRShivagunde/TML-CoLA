# Introduction

This code repo is mainly based on CoLA code repo with some additional features.

For **original CoLA codebase**, please refer to `https://github.com/alvin-zyl/CoLA`

`TML-CoLA` includes 
 - extra data processing scripts to support offline data access
 - has tensorboard support
 - warm_stable_decay lr scheduler support
 - has config files for model maintaining same aspect ratio (aspect ratio=64) Hidden size/ #heads ratio and Hidden size/ # layers for all model sizes, the config files changed from original config are named with `_aspectratio` tag e.g. `CoLA/baseline_configs/llama_350m_aspectratio64.json`


Note: 
 - `CoLA` original code https://github.com/alvin-zyl/CoLA  repo supports only bf16 precision training.
 - [THIS REPO]  `TML-CoLA` code repo https://github.com/NamrataRShivagunde/TML-CoLA supports only bf16 precision training.
 - `low-rank-training` code repo https://github.com/NamrataRShivagunde/low-rank-training  - implemented Cola in Nemo framework and supports mixed precision training. 

# TML-CoLA
cd TML-CoLA

    conda create --name=tml-cola python=3.11

    pip install -r requirements.txt

If datasets is not downloaded, go to download dataset section.

## To download data 

refer scripts/DOWNLOAD_C4_INSTRUCTIONS.md and follow instructions

```bash
    cd TML-CoLA
   
    python scripts/download_c4_offline.py --target_tokens 5000 --val_tokens 500 --output_dir ./datasets-test/c4/tokenized

    python scripts/download_c4_offline.py --target_tokens 3000000000 --val_tokens 10000000 --output_dir ./datasets-3B/c4/tokenized
```


# TML-CoLA
To run baseline, specify model config at e.g. CoLA/baseline_configs/llama_60m.json, 

Set hyperparameters by checking CoLA/main.py args and then run

    torchrun --standalone --nproc_per_node 1 CoLA/main.py \
        --model_config CoLA/baseline_configs/llama_60m.json \
        --model_type llama \
        --lr 0.001 \
        --batch_size 256 \
        --total_batch_size 512 \
        --num_training_steps 10000 \
        --warmup_steps 1000 \
        --weight_decay 0 \
        --dtype bfloat16 \
        --eval_every 1000 \
        --optimizer adamw \
        --run_name baseline-60m-cosine \
        --scheduler cosine \
        --offline_mode \
        --offline_data_path datasets/c4/tokenized \
        --tensorboard


## Cola
To run Cola, specify model config at  e,g. CoLA/cola_configs/cola_60m.json, 

Set hyperparameters by checking CoLA/main.py args and then run

    CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 CoLA/main.py \
        --model_type cola \
        --model_config CoLA/cola_configs/cola_60m.json \
        --lr 0.006 \
        --optimizer adamw \
        --batch_size 256 \
        --total_batch_size 512 \
        --num_training_steps 10000 \
        --warmup_steps 2000 \
        --weight_decay 0.01 \
        --dtype bfloat16 \
        --eval_every 1000 \
        --grad_clipping 0.5 \
        --run_name cola-60m-cosine





## miniconda

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh        
    
    bash ~/miniconda.sh

    conda create --name=tml-cola python=3.11
    
    conda activate tml-cola


# How to run baseline and cola for all model sizes

We have bash scripts in `bash_scripts` or we mentioned all of them below for easy access and changes.

# Baselines

## Baseline - 60M

    torchrun --standalone --nproc_per_node 1 CoLA/main.py \
        --model_config CoLA/baseline_configs/llama_60m.json \
        --model_type llama \
        --lr 0.001 \
        --batch_size 256 \
        --total_batch_size 512 \
        --num_training_steps 10000 \
        --warmup_steps 1000 \
        --weight_decay 0 \
        --dtype bfloat16 \
        --eval_every 1000 \
        --optimizer adamw \
        --run_name baseline-60m \
        --scheduler cosine \
        --offline_mode \
        --offline_data_path datasets/c4/tokenized \
        --tensorboard


## Baseline - 130M

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


## Baseline - 350M

    torchrun --standalone --nproc_per_node 1 main_withwandb.py \
        --model_config baseline_configs/llama_350m_aspectratio64.json \
        --model_type llama \
        --lr 0.0005 \
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

## Baseline - 1B 

    torchrun --standalone --nproc_per_node 8 CoLA/main.py \
        --model_config CoLA/baseline_configs/llama_1b_aspectratio64.json \
        --model_type llama \
        --lr 0.0005 \
        --batch_size 16 \
        --total_batch_size 512 \
        --num_training_steps 100000 \
        --warmup_steps 10000 \
        --weight_decay 0.0 \
        --dtype bfloat16 \
        --eval_every 1000 \
        --optimizer adamw \
        --run_name baseline-1B \
        --scheduler cosine \
        --offline_mode \
        --offline_data_path datasets/c4/tokenized \
        --tensorboard


# CoLA

## Cola - 60M

    torchrun --standalone --nproc-per-node=1 CoLA/main.py \
        --model_type cola \
        --model_config CoLA/cola_configs/cola_60m.json \
        --lr 0.006 \
        --optimizer adamw \
        --batch_size 256 \
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
        --offline_data_path datasets/c4/tokenized \
        --tensorboard

## Cola - 130M

    torchrun --standalone --nproc-per-node=1 CoLA/main.py \
        --model_type cola \
        --model_config CoLA/cola_configs/cola_130m.json \
        --lr 0.003 \
        --optimizer adamw \
        --batch_size 256 \
        --total_batch_size 512 \
        --num_training_steps 20000 \
        --warmup_steps 2000 \
        --weight_decay 0.01 \
        --dtype bfloat16 \
        --eval_every 1000 \
        --grad_clipping 0.5 \
        --run_name cola-130m \
        --scheduler cosine \
        --offline_mode \
        --offline_data_path datasets/c4/tokenized \
        --tensorboard


## Cola - 350M

    torchrun --standalone --nproc-per-node=1 CoLA/main.py \
        --model_type cola \
        --model_config CoLA/cola_configs/cola_350m_aspectratio64.json \
        --lr 0.003 \
        --optimizer adamw \
        --batch_size 64 \
        --total_batch_size 512 \
        --num_training_steps 60000 \
        --warmup_steps 6000 \
        --weight_decay 0.01 \
        --dtype bfloat16 \
        --eval_every 1000 \
        --grad_clipping 0.5 \
        --run_name cola-350m \
        --scheduler cosine \
        --offline_mode \
        --offline_data_path datasets/c4/tokenized \
        --tensorboard

## Cola - 1B

    torchrun --standalone --nproc_per_node 8 CoLA/main.py \
        --model_config  CoLA/cola_configs/cola_1b_aspectratio64.json \
        --model_type cola \
        --lr 0.002 \
        --batch_size 16 \
        --total_batch_size 512 \
        --num_training_steps 100000 \
        --warmup_steps 10000 \
        --weight_decay 0.01 \
        --grad_clipping 0.5 \
        --dtype bfloat16 \
        --eval_every 1000 \
        --optimizer adamw \
        --run_name cola-1b \
        --scheduler cosine \
        --offline_mode \
        --offline_data_path datasets/c4/tokenized \
        --tensorboard
