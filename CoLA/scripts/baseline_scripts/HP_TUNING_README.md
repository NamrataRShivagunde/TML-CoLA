# Hyperparameter Tuning Scripts

This directory contains scripts for automated hyperparameter tuning of the baseline LLaMA model.

## Scripts Overview

### `hp_tuning_gpu0.sh` - GPU 0
Tunes:
- **Learning Rate**: [0.0001, 0.0005, 0.001, 0.01, 0.05]
- **Warmup Steps**: [500, 1000, 2000, 3000, 4000]

Total runs: 10 (5 LR + 5 warmup)

### `hp_tuning_gpu1.sh` - GPU 1
Tunes:
- **Weight Decay**: [0.0, 0.005, 0.01, 0.05, 0.1]
- **Adam Epsilon**: [1e-14, 1e-8, 1e-6]

Total runs: 8 (5 WD + 3 epsilon)

## How to Run

### Run both scripts simultaneously on 2 GPUs:

```bash
cd /Users/nammu/code/TML-CoLA

# Terminal 1 (GPU 0)
bash CoLA/scripts/baseline_scripts/hp_tuning_gpu0.sh

# Terminal 2 (GPU 1) 
bash CoLA/scripts/baseline_scripts/hp_tuning_gpu1.sh
```

Or run them in the background:

```bash
cd /Users/nammu/code/TML-CoLA

# Run both in background
nohup bash CoLA/scripts/baseline_scripts/hp_tuning_gpu0.sh > gpu0_tuning.log 2>&1 &
nohup bash CoLA/scripts/baseline_scripts/hp_tuning_gpu1.sh > gpu1_tuning.log 2>&1 &

# Monitor progress
tail -f gpu0_tuning.log
tail -f gpu1_tuning.log
```

## Results

All results are automatically logged to:
```
./hp_tuning_results/hp_results.txt
```

Each line contains:
- run_name
- lr
- warmup_steps
- stable_steps
- weight_decay
- final_eval_loss
- final_eval_perplexity

## Configuration

Default base configuration (used when not tuning a specific parameter):
- Learning Rate: 0.001
- Warmup Steps: 2000
- Stable Steps: 7000
- Weight Decay: 0.01
- Batch Size: 128
- Total Batch Size: 512
- Training Steps: 10000
- Scheduler: warm_stable_decay

## Modifying Hyperparameter Values

To change the values being tested, edit the arrays in each script:

**hp_tuning_gpu0.sh:**
```bash
LR_VALUES=(0.0001 0.0005 0.001 0.01 0.05)
WARMUP_VALUES=(500 1000 2000 3000 4000)
```

**hp_tuning_gpu1.sh:**
```bash
WD_VALUES=(0.0 0.005 0.01 0.05 0.1)
ADAM_EPS_VALUES=(1e-14 1e-8 1e-6)
```

## Note on Adam Epsilon

Adam epsilon tuning currently uses the default epsilon value as the argument may need to be added to `main_withwandb.py`. If you want to tune this parameter, you may need to add `--adam_epsilon` argument support.

## Analyzing Results

After all runs complete, you can analyze results:

```bash
# View all results
cat ./hp_tuning_results/hp_results.txt

# Sort by eval loss (lower is better)
sort -t'=' -k6 -n ./hp_tuning_results/hp_results.txt

# Find best learning rate
grep "lr=" ./hp_tuning_results/hp_results.txt | sort -t'=' -k6 -n | head -5

# Find best warmup steps
grep "warm=" ./hp_tuning_results/hp_results.txt | sort -t'=' -k6 -n | head -5
```

## Total Time Estimate

- Each run: ~varies based on hardware
- GPU 0: 10 runs
- GPU 1: 8 runs
- Total: 18 runs

With 10,000 training steps each, estimate total time per GPU accordingly.
