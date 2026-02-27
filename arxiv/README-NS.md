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
    --offline_data_path datasets-3.0.0/c4/tokenized \
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


# offline mode test

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 CoLA/main_withwandb.py \
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
    --run_name baseline-60m-cosine-offline-mbs128--newenv-newdata \
    --scheduler cosine \
    --offline_mode \
    --offline_data_path datasets-1pt5B/c4/tokenized 


CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc-per-node=1 CoLA/main_withwandb.py \
    --model_type cola \
    --model_config CoLA/cola_configs/cola_60m.json \
    --lr 0.006 \
    --optimizer adamw \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name cola-60m-cosine-offline-mbs64-newenv-newdata \
    --scheduler cosine \
    --offline_mode \
    --offline_data_path datasets-1pt5B/c4/tokenized 


# online mode test

CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node 1 CoLA/main_withwandb.py \
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
    --run_name baseline-60m-cosine-offline-mbs128-newenv-newdata \
    --scheduler cosine 


CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc-per-node=1 CoLA/main_withwandb.py \
    --model_type cola \
    --model_config CoLA/cola_configs/cola_60m.json \
    --lr 0.006 \
    --optimizer adamw \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name cola-60m-cosine-offline-mbs64-newenv-newdata \
    --scheduler cosine


# compare datasets

Save data after dataloader Iterate over dataloader and then save

     python scripts/download_c4_offline.py --target_tokens 50000 --val_tokens 5000 --output_dir ./datasets/c4/tokenized_old

# optimized script, it is faster and works when script crashes and needs to resume
    
    python scripts/optimized_download_c4_offline_withresume.py \
        --target_tokens 50000 \                 
        --val_tokens 5000 \               
        --max_length 256 \
        --batch_size 512 \
        --output_dir datasets/c4/tokenized --shard_size 20

    python scripts/optimized_download_c4_offline_withresume.py --target_tokens 1500000000 --val_tokens 10000000 --max_length 256 --batch_size 512 --output_dir datasets-1pt5B/c4/tokenizer --shard_size 5000

# checked with `download_c4_offline.py` and `optimized_download_c4_offline_withresume.py` creates same datasets

python scripts/compare_c4_datasets.py \
    --old_path datasets/c4/tokenized_old \
    --new_path datasets/c4/tokenized


outputs

            === LOADING DATASETS ===
        Loaded both datasets.

        📌 METADATA COMPARISON
        - Old: {'tokenizer': 't5-base', 'max_length': 256, 'train_non_pad_tokens': 50156, 'train_examples': 269, 'val_non_pad_tokens': 5212, 'val_examples': 26}
        - New: {'tokenizer': 't5-base', 'max_length': 256, 'train_non_pad_tokens': 50156, 'train_examples': 269, 'val_non_pad_tokens': 5212, 'val_examples': 26}
        ✅ Metadata matches exactly.

        🔍 Comparing split: train
        - Old train length: 269
        - New train length: 269
        ➡️ Lengths match.
        ➡️ Computing example hashes...
        ✅ Split train is IDENTICAL!

        🔍 Comparing split: validation
        - Old validation length: 26
        - New validation length: 26
        ➡️ Lengths match.
        ➡️ Computing example hashes...
        ✅ Split validation is IDENTICAL!

        🎉🎉🎉 FINAL RESULT: DATASETS ARE 100% IDENTICAL 🎉🎉🎉



# precision

 float 32 everything

CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc-per-node=1 CoLA/main_withwandb.py \
    --model_type cola \
    --model_config CoLA/cola_configs/cola_60m.json \
    --lr 0.006 \
    --optimizer adamw \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype float32 \
    --eval_every 1000 \
    --grad_clipping 0.5 \
    --run_name cola-60m-cosine-fp32 \
    --scheduler cosine


# raw c4 to tokenized

python scripts/optimized_convert_raw_c4_to_tokenized_c4.py     --input_dir ~/code/TML-CoLA/datasets-c4-raw/c4     --output_dir datasets/c4/tokenized     --target_tokens 50000     --val_tokens 5000     --max_length 256     --batch_size 512     --shard_size 20

# testing
ran this to get dummy shards


python scripts/optimized_convert_raw_c4_to_tokenized_c4-NS.py     --input_dir ~/code/TML-CoLA/datasets-c4-raw/c4     --output_dir datasets-testing-nov26/c4/tokenized     --target_tokens 50000     --val_tokens 5000     --max_length 256     --batch_size 512     --shard_size 20

datasets-testing-nov26/c4/tokenized/ has shards

lets assemble shards 0 to 2

python scripts/assemble_train_shards.py \
    --shard_dir datasets-testing-nov26/c4/tokenized/ \
    --output_dir datasets-subset-nov26/c4/tokenized/train/ \
    --train_start 0 \
    --train_end 2

val set shard save and assemble

python scripts/optimized_convert_raw_c4_to_tokenized_c4_only_val_set.py \
    --input_dir ~/code/TML-CoLA/datasets-c4-raw/c4 \
    --output_dir datasets-subset-nov26/c4/tokenized/validation/ \
    --val_tokens 5000 \
    --max_length 256 \
    --batch_size 512 \
    --shard_size 1000


------

Hand over scripts

git pull

# to assemble subset of training shards
python scripts/assemble_train_shards.py \
    --shard_dir datasets-testing-nov26/c4/tokenized/ \
    --output_dir datasets-subset-nov26/c4/tokenized/ \
    --train_start 0 \
    --train_end 2

# to process val set separately as we won't have val shards yet
python scripts/optimized_convert_raw_c4_to_tokenized_c4_only_val_set.py \
    --input_dir ~/code/TML-CoLA/datasets-c4-raw/c4 \
    --output_dir datasets-subset-nov26/c4/tokenized/ \
    --val_tokens 10000000 \
    --max_length 256 \
    --batch_size 512 \
    --shard_size 5000


## OFFLINE
# baseline
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node 1 CoLA/main_withwandb.py \
    --model_config CoLA/baseline_configs/llama_60m.json \
    --model_type llama \
    --lr 0.001 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.0 \
    --grad_clipping 0.5 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adamw \
    --run_name baseline-60m-cosine-wd0.0-clipgrad0.5 \
    --scheduler cosine \
    --offline_mode \
    --offline_data_path datasets-1pt5B/c4/tokenizer/

# cola
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node 1 CoLA/main_withwandb.py \
    --model_config CoLA/baseline_configs/llama_60m.json \
    --model_type llama \
    --lr 0.001 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.0 \
    --grad_clipping 0.5 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adamw \
    --run_name baseline-60m-cosine-wd0.0-clipgrad0.5 \
    --scheduler cosine \
    --offline_mode \
    --offline_data_path datasets-1pt5B/c4/tokenizer/


# ONLINE

# saving at  
Saving model to checkpoints/llama_60m-2025-12-08-17-03-58 every 500 update steps

CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node 1 CoLA/main_withwandb.py \
    --model_config CoLA/baseline_configs/llama_60m.json \
    --model_type llama \
    --lr 0.001 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adamw \
    --run_name baseline-60m-losslandscape_online \
    --scheduler cosine \
    --save_every 500


# Cola
Saving model to checkpoints/cola_60m-2025-12-08-17-12-58 every 500 update steps
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 CoLA/main_withwandb.py \
    --model_config CoLA/cola_configs/cola_60m.json \
    --model_type cola \
    --lr 0.006 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --grad_clipping 0.5 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adamw \
    --run_name cola-60m-losslandscape_online \
    --scheduler cosine \
    --save_every 500


# online mbs 64 as mbs 128 spikes for online cola
 Saving model to checkpoints/cola_60m-2025-12-08-20-59-46 every 500 update steps

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 CoLA/main_withwandb.py     --model_config CoLA/cola_configs/cola_60m.json     --model_type cola     --lr 0.006     --batch_size 64     --total_batch_size 512     --num_training_steps 10000     --warmup_steps 2000     --weight_decay 0.01     --grad_clipping 0.5     --dtype bfloat16     --eval_every 1000     --optimizer adamw     --run_name cola-60m-losslandscape_online_mbs64    --scheduler cosine     --save_every 500

Saving model to checkpoints/llama_60m-2025-12-08-21-00-08 every 500 update steps
CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node 1 CoLA/main_withwandb.py     --model_config CoLA/baseline_configs/llama_60m.json     --model_type llama     --lr 0.001     --batch_size 64     --total_batch_size 512     --num_training_steps 10000     --warmup_steps 1000     --weight_decay 0.0     --dtype bfloat16     --eval_every 1000     --optimizer adamw     --run_name baseline-60m-losslandscape_online_mbs64     --scheduler cosine     --save_every 500