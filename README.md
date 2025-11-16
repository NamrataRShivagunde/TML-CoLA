# TML-CoLA
cd TML-CoLA

conda create --name=tml-cola python=3.11

pip install -r requirements.txt



## Baseline
To run baseline, specify model config at CoLA/baseline_configs/llama_60m.json, 

Set hyperparameters by checking CoLA/main.py args and then

torchrun --standalone --nproc_per_node 1 CoLA/main.py \
    --model_config CoLA/baseline_configs/llama_60m.json \
    --model_type llama \
    --lr 0.001 \
    --batch_size 128 \
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


To run Cola, specify model config at CoLA/cola_configs/cola_60m.json, 

Set hyperparameters by checking CoLA/main.py args and then

    cd CoLA


     bash scripts/cola_scripts/cola60m_offline.sh


## to download data first, 
    refer scripts/DOWNLOAD_C4_INSTRUCTIONS.md and follow instructions

```bash
cd /Users/nammu/code/TML-CoLA
    python scripts/download_c4_offline.py --target_tokens 50000 --val_tokens 5000 --output_dir ./datasets/c4/tokenized
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