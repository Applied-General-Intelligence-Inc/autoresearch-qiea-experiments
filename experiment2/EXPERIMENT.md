# QIEA-Guided Neural Architecture Search for AutoResearch

This document describes experiment2, where QIEA performs binary structural search over `train.py` architecture decisions, and the AutoResearch LLM applies those decisions each run.

Primary optimization target: **`val_bpb`** (lower is better) after the fixed 5-minute training window.

## Purpose

Instead of searching only continuous optimizer hyperparameters, this experiment searches **architecture structure** using q-bits:

- Norm family: RMSNorm vs LayerNorm
- Attention regime: full-context standard attention vs sliding-window attention
- MLP family: ReLU^2 MLP vs SwiGLU MLP
- Value embeddings pathway: off vs on
- Q/K normalization in attention: off vs on
- RoPE frequency base: 10k vs 50k

Each run samples a binary architecture string, writes a plaintext blueprint, and forces the agent to implement it in `train.py` before training.

## Technical Flow

1. `qiea_optimizer.py sample`
   - Collapses architecture q-bits into a binary string (example: `010011`).
   - Writes:
     - `architecture_blueprint.txt` (human/agent-readable plaintext)
     - `architecture_blueprint.json` (structured metadata)
   - Stores pending observation in `qiea_state.json`.

2. Agent reads `architecture_blueprint.txt`
   - Updates only the `QIEA_ARCH_BLUEPRINT` constant block in `train.py`.
   - Runs compile sanity check: `uv run python -m py_compile train.py`.

3. Run training
   - `uv run train.py > run.log 2>&1`
   - `train.py` executes for fixed 5 minutes and prints summary with `val_bpb`.
   - `train.py` writes `qiea_last_result.json`.

4. Fitness update
   - `qiea_evaluate.py` resolves `val_bpb` from CLI/result/log.
   - Collects commit diff diversity for `train.py` using `git diff HEAD~1..HEAD --numstat`.
   - Calls `qiea_optimizer.update_from_result(...)`.
   - QIEA applies rotation-gate updates:
     - if improved: rotate toward sampled bits
     - if worse: rotate away from sampled bits

## Files Produced

Core state/artifacts:

- `qiea_state.json` - q-bit amplitudes, pending observation, best score
- `architecture_blueprint.txt` - plaintext architecture instructions for the agent
- `architecture_blueprint.json` - binary string and constant assignments
- `qiea_last_result.json` - final run metrics from `train.py`

Tracked metrics:

- `qiea_convergence.csv` - run-level `val_bpb`, improvement flag, best-so-far, binary string
- `qiea_trajectories.csv` - per-qbit amplitude trajectory (`alpha`, `beta`, `P(bit=1)`)
- `qiea_architecture_frequency.csv` - sampled architecture choice frequencies over runs
- `qiea_diff_diversity.csv` - commit-level change diversity in `train.py`

## How To Run

## Prerequisites

- NVIDIA GPU
- Python 3.10+
- `uv` installed
- Dependencies installed in this folder (`uv sync`)
- Data/tokenizer cache exists (`uv run prepare.py` once)

## 1) Initialize QIEA architecture state

```bash
cd experiment2
uv run qiea_optimizer.py init --force
```

## 2) Sample a new architecture blueprint

```bash
uv run qiea_optimizer.py sample
cat architecture_blueprint.txt
```

## 3) Apply blueprint to train.py

- Open `train.py`.
- Update constants between:
  - `# BEGIN QIEA_ARCH_BLUEPRINT`
  - `# END QIEA_ARCH_BLUEPRINT`
- Use exact assignments from `architecture_blueprint.txt`.

Sanity check:

```bash
uv run python -m py_compile train.py
```

## 4) Run one 5-minute experiment

```bash
uv run train.py > run.log 2>&1
```

Extract metrics:

```bash
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## 5) Update QIEA from fitness

```bash
uv run qiea_evaluate.py --run-log run.log
```

## 6) Inspect state and convergence

```bash
uv run qiea_optimizer.py status
```

Then inspect CSV artifacts listed above.

## Automated 100-Run Execution with compare.py

If you want to automate the full QIEA loop (sample -> apply blueprint -> compile check -> train -> evaluate/update) repeatedly, use `compare.py`.

This script only runs QIEA experiments in this folder. It does **not** launch any baseline/original runs.

Run 100 QIEA experiments (default):

```bash
cd experiment2
uv run compare.py
```

Explicit run count:

```bash
uv run compare.py --runs 100
```

Resume after interruption:

```bash
uv run compare.py --runs 100 --resume
```

Use existing baseline CSV for final summary-only comparison (no baseline execution):

```bash
uv run compare.py --runs 100 --baseline-csv ../original/bench/original_results.csv
```

Notes:

- QIEA result rows are written to `bench/qiea_results.csv`.
- Per-run logs are written to `bench/qiea_run_<i>.log` and `bench/qiea_update_<i>.log`.
- If a run crashes, `compare.py` records a crash row and still applies a penalty fitness update so the q-bit state can continue evolving.

## Suggested Autonomous Loop

Use `program.md` as the operating policy for the agent. The intended loop is:

1. Sample q-bits
2. Apply blueprint edits to `train.py`
3. Commit
4. Train
5. Evaluate/update q-bits
6. Log keep/discard decision
7. Repeat

## Interpretation

- Lower `val_bpb` indicates better architecture under fixed-time compute.
- `qiea_architecture_frequency.csv` shows which structural choices dominate over time (for example, SwiGLU vs ReLU^2).
- `qiea_diff_diversity.csv` quantifies how radically `train.py` changes between commits, ensuring architectural exploration remains active.

## References

- AutoResearch repository and metric definition: https://github.com/karpathy/autoresearch
- PyTorch module construction: https://pytorch.org/docs/stable/nn.html
- NAS concepts background: https://arxiv.org/abs/1808.05377
