# QIEA vs Original AutoResearch: Reproducible Experiment Guide

This document explains exactly how to run and compare:

- Original AutoResearch from `karpathy/autoresearch` (baseline)
- This QIEA-integrated version (Q-bit hyperparameter selection)

The primary metric is `val_bpb` (lower is better).

## What This Repo Adds

Compared to original AutoResearch, this repo adds:

- `qiea_optimizer.py` to initialize q-bits, collapse to `hyperparams.json`, and update amplitudes
- `qiea_evaluate.py` to apply post-run rotation-gate updates from final `val_bpb`
- `compare.py` to run baseline + QIEA benchmark loops and print the final comparison summary
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

- `autoresearch-original`
- `autoresearch-qiea`

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

## Step 4: Run Baseline + QIEA + Summary with compare.py

Use `compare.py` from the QIEA repo. It performs everything that was previously manual in Steps 4-6:

1. Baseline loop in `autoresearch-original`
2. QIEA loop in `autoresearch-qiea` (including sample -> train -> evaluate)
3. Final comparison summary printout

```bash
cd ~/experiments/autoresearch-compare/autoresearch-qiea
uv run compare.py --original-repo ../autoresearch-original --runs "$N"
```

By default, this command also resets QIEA state (`qiea_optimizer.py init --force`) so each benchmark starts clean.

If you intentionally want to continue from the existing QIEA state, add:

```bash
--skip-qiea-init
```

If a long benchmark is interrupted and you want to continue from existing CSV/state, use:

```bash
uv run compare.py --original-repo ../autoresearch-original --runs "$N" --resume --skip-qiea-init
```

Files written by `compare.py`:

- `../autoresearch-original/bench/original_results.csv`
- `../autoresearch-original/bench/original_run_<i>.log`
- `bench/qiea_results.csv`
- `bench/qiea_run_<i>.log`
- `bench/qiea_update_<i>.log`

## Step 5: Optional Interleaved Protocol (Stronger Fairness)

If you want to reduce time-drift effects (temperature, background load), use interleaved mode:

```bash
cd ~/experiments/autoresearch-compare/autoresearch-qiea
uv run compare.py \
  --original-repo ../autoresearch-original \
  --runs "$N" \
  --protocol interleaved
```

This runs paired iterations:

1. Original run `i`
2. QIEA run `i`

## Step 6: QIEA Artifacts Produced During Benchmark

QIEA-specific artifacts produced during this run:

- `hyperparams.json`
- `qiea_state.json`
- `qiea_convergence.csv`
- `qiea_trajectories.csv`

## Step 7: Generate Progress Graphs (Like progress.png)

After your benchmark finishes, generate progress-style graphs directly from the CSV results:

```bash
cd ~/experiments/autoresearch-compare/autoresearch-qiea
uv run plot_compare.py
```

This writes:

- `bench/progress_original.png`
- `bench/progress_qiea.png`
- `bench/compare_running_best.png`

To focus on the first `N` successful runs (for example, `N=100`):

```bash
uv run plot_compare.py --runs 100
```

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

When using `compare.py`, this order is handled automatically.

## Minimal One-Run Smoke Test

Before full benchmarking:

```bash
cd ~/experiments/autoresearch-compare/autoresearch-qiea
uv run compare.py --original-repo ../autoresearch-original --runs 1
```

Then inspect the two CSV files for `status=ok` rows:

```bash
cat ../autoresearch-original/bench/original_results.csv
cat bench/qiea_results.csv
```

If both runs succeed, proceed to the full `N`-run comparison.
