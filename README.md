# TML-CoLA

conda create --name=tml-cola python=3.11

pip install -r requirements.txt



## Baseline
To run baseline, specify model config at CoLA/baseline_configs/llama_60m.json, 

Set hyperparameters by checking CoLA/main.py args and then


run CoLA/scripts/baseline_scripts/baseline60m.sh

        bash scripts/baseline_scripts/baseline60m.sh/baseline60m.sh


## Cola


To run Cola, specify model config at CoLA/cola_configs/cola_60m.json, 

Set hyperparameters by checking CoLA/main.py args and then


run CoLA/scripts/cola_scripts/cola60m.sh

     bash /scripts/cola_scripts/cola60m.sh/cola60m.sh


## to download data first, 
    refer scripts/DOWNLOAD_C4_INSTRUCTIONS.md and follow instructions

```bash
cd /Users/nammu/code/TML-CoLA
python scripts/download_c4_offline.py --target_tokens 50000 --val_tokens 5000 --output_dir ./datasets/c4/tokenized
```

