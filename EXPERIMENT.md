# QIEA vs Original AutoResearch: Reproducible Experiment Guide

This document explains exactly how to run and compare:

- Original AutoResearch from `karpathy/autoresearch` (baseline)
- This QIEA-integrated version (Q-bit hyperparameter selection)

The primary metric is `val_bpb` (lower is better).

## What This Repo Adds

Compared to original AutoResearch, this repo adds:

- `qiea_optimizer.py` to initialize q-bits, collapse to `hyperparams.json`, and update amplitudes
- `qiea_evaluate.py` to apply post-run rotation-gate updates from final `val_bpb`
- `train.py` integration to load 10 continuous hyperparameters from `hyperparams.json`

The 10 continuous hyperparameters controlled by QIEA are:

1. `embedding_lr`
2. `unembedding_lr`
3. `matrix_lr`
4. `scalar_lr`
5. `weight_decay`
6. `adam_beta1`
7. `adam_beta2`
8. `warmup_ratio`
9. `warmdown_ratio`
10. `dropout`

## Prerequisites

- Single NVIDIA GPU
- Python 3.10+
- `uv` installed
- `git` installed

Use the same machine, GPU, driver, and CUDA stack for both variants.

## Recommended Directory Layout

Use two separate directories:

- `autoresearch-original` (baseline upstream)
- `autoresearch-qiea` (this repo)

Example:

```bash
mkdir -p ~/experiments/autoresearch-compare
cd ~/experiments/autoresearch-compare

# Clone original upstream baseline
git clone https://github.com/karpathy/autoresearch.git autoresearch-original

# Put this repo here as autoresearch-qiea (choose one approach):
# 1) Clone your fork/remote:
# git clone <your-qiea-repo-url> autoresearch-qiea
# 2) Copy your local working tree:
# cp -R /path/to/autoresearch-qiea-experiments autoresearch-qiea
```

## Step 1: Install Dependencies in Both Repos

```bash
cd ~/experiments/autoresearch-compare/autoresearch-original
uv sync

cd ~/experiments/autoresearch-compare/autoresearch-qiea
uv sync
```

## Step 2: Ensure Data/Tokenizer Exist

Run once (either repo is usually enough because cache is shared under `~/.cache/autoresearch/`):

```bash
cd ~/experiments/autoresearch-compare/autoresearch-original
uv run prepare.py
```

If you want to verify from both repos:

```bash
cd ~/experiments/autoresearch-compare/autoresearch-qiea
uv run prepare.py
```

## Step 3: Choose Experiment Length

Set number of runs. A practical comparison is `N=20`. For a long overnight run, use `N=100`.

```bash
N=20
```

## Step 4: Run Baseline (Original AutoResearch)

This runs original `train.py` repeatedly and stores metrics to CSV.

```bash
cd ~/experiments/autoresearch-compare/autoresearch-original
mkdir -p bench

printf "run,val_bpb,peak_vram_mb,training_seconds,total_seconds,status\n" > bench/original_results.csv

for i in $(seq 1 "$N"); do
  log="bench/original_run_${i}.log"
  uv run train.py > "$log" 2>&1

  val=$(grep '^val_bpb:' "$log" | awk '{print $2}')
  vram=$(grep '^peak_vram_mb:' "$log" | awk '{print $2}')
  train_s=$(grep '^training_seconds:' "$log" | awk '{print $2}')
  total_s=$(grep '^total_seconds:' "$log" | awk '{print $2}')

  if [ -z "$val" ]; then
    printf "%d,0.000000,0.0,0.0,0.0,crash\n" "$i" >> bench/original_results.csv
  else
    printf "%d,%s,%s,%s,%s,ok\n" "$i" "$val" "$vram" "$train_s" "$total_s" >> bench/original_results.csv
  fi

done
```

## Step 5: Run QIEA Variant (This Repo)

This loop does exactly one QIEA cycle per run:

1. Collapse q-bits to `hyperparams.json`
2. Run 5-minute training
3. Update q-bits from final `val_bpb`

```bash
cd ~/experiments/autoresearch-compare/autoresearch-qiea
mkdir -p bench

# Start from a clean QIEA state for a fresh benchmark
uv run qiea_optimizer.py init --force

printf "run,val_bpb,peak_vram_mb,training_seconds,total_seconds,status\n" > bench/qiea_results.csv

for i in $(seq 1 "$N"); do
  train_log="bench/qiea_run_${i}.log"
  update_log="bench/qiea_update_${i}.log"

  uv run qiea_optimizer.py sample
  uv run train.py > "$train_log" 2>&1
  uv run qiea_evaluate.py --run-log "$train_log" > "$update_log" 2>&1

  val=$(grep '^val_bpb:' "$train_log" | awk '{print $2}')
  vram=$(grep '^peak_vram_mb:' "$train_log" | awk '{print $2}')
  train_s=$(grep '^training_seconds:' "$train_log" | awk '{print $2}')
  total_s=$(grep '^total_seconds:' "$train_log" | awk '{print $2}')

  if [ -z "$val" ]; then
    printf "%d,0.000000,0.0,0.0,0.0,crash\n" "$i" >> bench/qiea_results.csv
  else
    printf "%d,%s,%s,%s,%s,ok\n" "$i" "$val" "$vram" "$train_s" "$total_s" >> bench/qiea_results.csv
  fi

done
```

QIEA-specific artifacts produced during this run:

- `hyperparams.json`
- `qiea_state.json`
- `qiea_convergence.csv`
- `qiea_trajectories.csv`

## Step 6: Compare Baseline vs QIEA

Run this from any directory after both loops complete:

```bash
uv run python - <<'PY'
import csv
import statistics
from pathlib import Path

orig = Path('~/experiments/autoresearch-compare/autoresearch-original/bench/original_results.csv').expanduser()
qiea = Path('~/experiments/autoresearch-compare/autoresearch-qiea/bench/qiea_results.csv').expanduser()

def load_rows(path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r['status'] == 'ok':
                rows.append(r)
    return rows

def vals(rows, key):
    return [float(r[key]) for r in rows]

def cum_best(xs):
    out = []
    best = None
    for x in xs:
        best = x if best is None else min(best, x)
        out.append(best)
    return out

orig_rows = load_rows(orig)
qiea_rows = load_rows(qiea)

orig_val = vals(orig_rows, 'val_bpb')
qiea_val = vals(qiea_rows, 'val_bpb')

orig_best_curve = cum_best(orig_val)
qiea_best_curve = cum_best(qiea_val)

print('Comparison summary')
print(f'  original_runs_ok: {len(orig_rows)}')
print(f'  qiea_runs_ok:     {len(qiea_rows)}')
print(f'  original_best:    {min(orig_val):.6f}')
print(f'  qiea_best:        {min(qiea_val):.6f}')
print(f'  delta_best:       {min(orig_val)-min(qiea_val):+.6f} (positive favors QIEA)')
print(f'  original_mean:    {statistics.mean(orig_val):.6f}')
print(f'  qiea_mean:        {statistics.mean(qiea_val):.6f}')
print(f'  original_median:  {statistics.median(orig_val):.6f}')
print(f'  qiea_median:      {statistics.median(qiea_val):.6f}')

k = min(10, len(orig_best_curve), len(qiea_best_curve))
if k > 0:
    print(f'  best_after_{k}_runs_original: {orig_best_curve[k-1]:.6f}')
    print(f'  best_after_{k}_runs_qiea:     {qiea_best_curve[k-1]:.6f}')
PY
```

## Optional: Interleaved Protocol (Stronger Fairness)

If you want to reduce time-drift effects (temperature, background load), run paired iterations:

1. Original run `i`
2. QIEA run `i`

Repeat until `N` is reached. Keep logging to the same CSV files.

## How to Interpret Results

- Lower `val_bpb` is better.
- Use both:
  - Final best value after `N` runs
  - Convergence speed (how quickly cumulative best improves)
- Check `peak_vram_mb` to ensure QIEA gains are not caused by unacceptable memory growth.

## Troubleshooting

- If `val_bpb` is missing in a log, treat that run as crash and inspect tail:

```bash
tail -n 80 <logfile>
```

- If QIEA update fails with missing pending observation, make sure each run includes:

1. `uv run qiea_optimizer.py sample`
2. `uv run train.py ...`
3. `uv run qiea_evaluate.py --run-log ...`

in that order.

## Minimal One-Run Smoke Test

Before full benchmarking:

```bash
# Original
cd ~/experiments/autoresearch-compare/autoresearch-original
uv run train.py > smoke_original.log 2>&1

grep '^val_bpb:\|^peak_vram_mb:' smoke_original.log

# QIEA
cd ~/experiments/autoresearch-compare/autoresearch-qiea
uv run qiea_optimizer.py init --force
uv run qiea_optimizer.py sample
uv run train.py > smoke_qiea.log 2>&1
uv run qiea_evaluate.py --run-log smoke_qiea.log

grep '^val_bpb:\|^peak_vram_mb:' smoke_qiea.log
```

If both smoke tests succeed, proceed to the full `N`-run comparison.
