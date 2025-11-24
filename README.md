# TML-CoLA
cd TML-CoLA

conda create --name=tml-cola python=3.11

pip install -r requirements.txt

If datasets is not downloaded, go to download dataset section.

## Baseline
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


## To download data 
    refer scripts/DOWNLOAD_C4_INSTRUCTIONS.md and follow instructions

```bash
cd /Users/nammu/code/TML-CoLA
    python scripts/download_c4_offline.py --target_tokens 5000 --val_tokens 500 --output_dir ./datasets/c4/tokenized

Dataset issue
https://github.com/huggingface/lerobot/issues/1538 
v3.x.x
https://github.com/huggingface/datasets/blob/3.6.0/src/datasets/features/features.py#L1418 
v4.x.x
https://github.com/huggingface/datasets/blob/4.4.1/src/datasets/features/features.py#L1418 

```

## miniconda

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh        
    
    bash ~/miniconda.sh

    conda create --name=tml-cola python=3.11
    
    conda activate tml-cola






CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node 1 main_withwandb.py \
    --model_config baseline_configs/llama_60m.json \
    --model_type llama \
    --lr 0.001 \
    --batch_size 128 \
    --total_batch_size 1024 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adamw \
    --run_name baseline-60m-wsd-gbs1024 \
    --scheduler warm_stable_decay


### 130M

CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc-per-node=1 CoLA/main_withwandb.py \
    --model_type cola \
    --model_config CoLA/cola_configs/cola_130m.json \
    --lr 0.003 \
    --optimizer adamw \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 1 \
    --run_name cola-130m-wsd-initA0.5-B0.7-clipgrad1-stable3k \
    --save_every 30000 \
    --scheduler warm_stable_decay \
    --stable_steps 3000