# CoLA Hyperparameter Tuning Scripts

Automated hyperparameter tuning for CoLA-60M model across 4 GPUs.

## Overview

### Script Distribution

| Script | GPU | Parameters Tuned | # Runs |
|--------|-----|------------------|--------|
| `cola_hp_tuning_gpu0.sh` | GPU 0 | Learning Rate, Warmup Steps | 10 |
| `cola_hp_tuning_gpu1.sh` | GPU 1 | Stable Steps, Weight Decay | 10 |
| `cola_hp_tuning_gpu2.sh` | GPU 2 | Gradient Clipping | 5 |
| `cola_hp_tuning_gpu3.sh` | GPU 3 | Combined Configs | 5 |
| **Total** | | | **30 runs** |

## Hyperparameter Values Tested

### GPU 0: Learning Rate & Warmup Steps
- **Learning Rate**: [0.003, 0.005, 0.006, 0.008, 0.01]
  - Range chosen for CoLA's low-rank structure (typically higher than baseline)
- **Warmup Steps**: [1000, 1500, 2000, 2500, 3000]
  - Testing 10%-30% of total steps

### GPU 1: Stable Steps & Weight Decay
- **Stable Steps**: [5000, 6000, 6500, 7000, 7500]
  - Testing 50%-75% stable phase ratios
- **Weight Decay**: [0.0, 0.005, 0.01, 0.02, 0.05]
  - From no regularization to strong regularization

### GPU 2: Gradient Clipping
- **Grad Clipping**: [0.0, 0.3, 0.5, 1.0, 2.0]
  - Testing no clipping to aggressive clipping

### GPU 3: Combined Configurations
5 hand-picked combinations based on typical successful patterns:
1. High LR (0.008) + High WD (0.02)
2. Lower LR (0.003) + Longer Warmup (3000)
3. Medium LR (0.006) + Longer Stable (7000) + Higher Clip (1.0)
4. High LR (0.01) + No WD (0.0) + Lower Clip (0.3)
5. Conservative: Lower everything for stability

## Default Base Configuration

When not tuning a specific parameter, these defaults are used:
```bash
Learning Rate:     0.006
Warmup Steps:      2000
Stable Steps:      6000
Weight Decay:      0.01
Gradient Clipping: 0.5
Total Steps:       10000
Batch Size:        128
Total Batch Size:  512
Scheduler:         warm_stable_decay
```

## How to Run

### Run All Scripts Simultaneously (4 GPUs)

```bash
cd /Users/nammu/code/TML-CoLA/CoLA

# Terminal 1 - GPU 0
bash scripts/cola_scripts/cola_hp_tuning_gpu0.sh

# Terminal 2 - GPU 1
bash scripts/cola_scripts/cola_hp_tuning_gpu1.sh

# Terminal 3 - GPU 2
bash scripts/cola_scripts/cola_hp_tuning_gpu2.sh

# Terminal 4 - GPU 3
bash scripts/cola_scripts/cola_hp_tuning_gpu3.sh
```

### Run in Background

```bash
cd /Users/nammu/code/TML-CoLA/CoLA

nohup bash scripts/cola_scripts/cola_hp_tuning_gpu0.sh > cola_gpu0.log 2>&1 &
nohup bash scripts/cola_scripts/cola_hp_tuning_gpu1.sh > cola_gpu1.log 2>&1 &
nohup bash scripts/cola_scripts/cola_hp_tuning_gpu2.sh > cola_gpu2.log 2>&1 &
nohup bash scripts/cola_scripts/cola_hp_tuning_gpu3.sh > cola_gpu3.log 2>&1 &

# Monitor progress
tail -f cola_gpu0.log
```

## Results

All results are saved to:
```
./cola_hp_tuning_results/hp_results.txt
```

Each line contains:
```
run_name=cola-60m-wsd-init-scalept5-..., lr=X, warmup_steps=Y, stable_steps=Z, weight_decay=W, grad_clipping=G, final_eval_loss=L, final_eval_perplexity=P
```

## Analyzing Results

### View all results
```bash
cat ./cola_hp_tuning_results/hp_results.txt
```

### Sort by best eval loss
```bash
sort -t'=' -k7 -n ./cola_hp_tuning_results/hp_results.txt | grep -v "FAILED" | head -10
```

### Find best for each hyperparameter
```bash
# Best learning rate
grep "lr=" ./cola_hp_tuning_results/hp_results.txt | sort -t'=' -k7 -n | head -5

# Best warmup steps
grep "warm=" ./cola_hp_tuning_results/hp_results.txt | sort -t'=' -k7 -n | head -5

# Best stable steps
grep "stable=" ./cola_hp_tuning_results/hp_results.txt | sort -t'=' -k7 -n | head -5

# Best weight decay
grep "wd=" ./cola_hp_tuning_results/hp_results.txt | sort -t'=' -k7 -n | head -5

# Best gradient clipping
grep "clipgrad=" ./cola_hp_tuning_results/hp_results.txt | sort -t'=' -k7 -n | head -5
```

### Statistics
```bash
# Count successful vs failed runs
echo "Successful: $(grep -v "FAILED" ./cola_hp_tuning_results/hp_results.txt | wc -l)"
echo "Failed: $(grep "FAILED" ./cola_hp_tuning_results/hp_results.txt | wc -l)"

# Average eval loss of successful runs
grep -v "FAILED" ./cola_hp_tuning_results/hp_results.txt | \
  awk -F'final_eval_loss=' '{print $2}' | \
  awk -F',' '{sum+=$1; count++} END {print "Average:", sum/count}'
```

## Stopping and Resuming

All scripts include error handling:
- ✅ Stop via wandb UI → continues to next run
- ✅ Ctrl+C → continues to next run  
- ✅ Failed runs logged as "FAILED"
- ✅ See `STOPPING_RUNS_GUIDE.md` for details

## Modifying Values

To test different hyperparameter values, edit the arrays in each script:

**Example - GPU 0 (`cola_hp_tuning_gpu0.sh`):**
```bash
# Change learning rates
LR_VALUES=(0.003 0.005 0.006 0.008 0.01)  # ← Edit here

# Change warmup steps
WARMUP_VALUES=(1000 1500 2000 2500 3000)  # ← Edit here
```

## Run Name Convention

All runs follow this naming pattern:
```
cola-60m-wsd-init-scalept5-lr{VALUE}-warm{VALUE}-stable{VALUE}-decay{VALUE}-[wd{VALUE}]-[clipgrad{VALUE}]
```

- `cola-60m`: Model size
- `wsd`: Warm-Stable-Decay scheduler
- `init-scalept5`: Identifier for this tuning run (scale point 5)
- Individual HP values clearly labeled

## Time Estimates

- **Per run**: ~varies by hardware (10k steps)
- **GPU 0**: 10 runs
- **GPU 1**: 10 runs  
- **GPU 2**: 5 runs
- **GPU 3**: 5 runs
- **Total**: 30 runs across 4 GPUs

If running in parallel on 4 GPUs, expect roughly the time of the longest script (GPU 0 or GPU 1).

## Tips

1. **Start with one GPU first** to validate everything works
2. **Monitor early runs** in wandb to catch issues
3. **Check logs regularly**: `tail -f cola_gpu*.log`
4. **Failed runs can be rerun** individually (see STOPPING_RUNS_GUIDE.md)
5. **Compare with baseline** results from `hp_tuning_results/hp_results.txt`

## Next Steps After Tuning

1. Identify best configuration from results
2. Run 3-5 seeds with best config for statistical significance
3. Scale to larger model sizes (130M, 350M, etc.) with similar HPs
4. Document findings in your results

## Wandb Integration

All runs automatically log to wandb with:
- Training loss curves
- Learning rate schedule
- Gradient norms
- Memory usage
- Throughput metrics
- Final evaluation metrics

Group/filter runs by the `init-scalept5` tag to analyze this sweep.
