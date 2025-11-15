# How to Stop a Run and Continue

## Stopping Runs

The scripts now have error handling that allows them to continue even if a run is stopped or fails.

### Option 1: Stop via Wandb UI

1. Go to your wandb run page
2. Click the "Stop Run" button
3. The current run will terminate
4. **The script will automatically continue to the next hyperparameter configuration**
5. The failed run will be logged as "FAILED" in `hp_results.txt`

### Option 2: Stop via Ctrl+C (Terminal)

If you press Ctrl+C in the terminal:
- The current run will stop
- The script will catch the error and continue to the next run
- A "FAILED" entry will be added to the results file

### Option 3: Kill Specific Run via Process

```bash
# Find the process
ps aux | grep main_withwandb.py

# Kill specific process
kill <PID>

# The script will continue to the next run
```

## What Happens When You Stop

When a run is stopped (by any method):

1. ✅ Script prints: `WARNING: Run <name> failed or was interrupted! Continuing to next run...`
2. ✅ Logs a FAILED entry to `hp_results.txt`
3. ✅ Automatically starts the next hyperparameter configuration
4. ✅ Other GPU's script continues unaffected

Example failed entry in `hp_results.txt`:
```
run_name=baseline-60m-wsd-lr0.01-warm2000-decay1000-stable7000, lr=0.01, warmup_steps=2000, stable_steps=7000, weight_decay=0.01, final_eval_loss=FAILED, final_eval_perplexity=FAILED
```

## Monitoring Runs

### Real-time monitoring:

```bash
# Watch the progress
tail -f gpu0_tuning.log
tail -f gpu1_tuning.log

# Check results file
tail -f ./hp_tuning_results/hp_results.txt
```

### Check which runs completed successfully:

```bash
# Show successful runs
grep -v "FAILED" ./hp_tuning_results/hp_results.txt

# Show failed runs
grep "FAILED" ./hp_tuning_results/hp_results.txt

# Count successful vs failed
echo "Successful: $(grep -v "FAILED" ./hp_tuning_results/hp_results.txt | wc -l)"
echo "Failed: $(grep "FAILED" ./hp_tuning_results/hp_results.txt | wc -l)"
```

## Rerunning Failed Configurations

If you want to rerun a specific configuration that failed:

```bash
# Example: Rerun a specific config
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 CoLA/main_withwandb.py \
    --model_config CoLA/baseline_configs/llama_60m.json \
    --model_type llama \
    --lr 0.01 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 2000 \
    --stable_steps 7000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 20000 \
    --optimizer adamw \
    --scheduler warm_stable_decay \
    --run_name baseline-60m-wsd-lr0.01-warm2000-decay1000-stable7000 \
    --save_dir ./hp_tuning_results \
    --single_gpu
```

## Stopping ALL Runs Completely

If you want to stop the entire hyperparameter search:

### Method 1: Stop the background process

```bash
# Find the script process
ps aux | grep hp_tuning_gpu

# Kill the script
kill <PID>
```

### Method 2: Create a stop file

Add this to your script workflow:

```bash
# In terminal, create a stop signal
touch ./hp_tuning_results/STOP

# Then modify your loop to check for it (optional enhancement)
```

## Best Practices

1. **Monitor Early**: Watch the first few runs to ensure everything works
2. **Check Spikes**: If you see a loss spike, you can stop and the next run continues
3. **Review Logs**: The `gpu0_tuning.log` and `gpu1_tuning.log` files show all output
4. **Partial Results**: Even if some runs fail, you'll have results for successful ones

## Example: Full Workflow

```bash
# Start both scripts
nohup bash CoLA/scripts/baseline_scripts/hp_tuning_gpu0.sh > gpu0_tuning.log 2>&1 &
nohup bash CoLA/scripts/baseline_scripts/hp_tuning_gpu1.sh > gpu1_tuning.log 2>&1 &

# Monitor
tail -f gpu0_tuning.log

# See a spike in wandb UI? Stop the run there
# Script automatically continues to next config

# Check progress
grep -c "Completed" gpu0_tuning.log
grep -c "FAILED" ./hp_tuning_results/hp_results.txt

# When done, analyze results
sort -t'=' -k6 -n ./hp_tuning_results/hp_results.txt | grep -v "FAILED" | head -5
```

## Summary

✅ **You CAN stop a run via wandb UI** - the script will continue  
✅ **You CAN Ctrl+C a run** - the script will continue  
✅ Failed runs are logged as "FAILED" in the results  
✅ Successful runs are logged normally  
✅ No need to restart the entire hyperparameter sweep
