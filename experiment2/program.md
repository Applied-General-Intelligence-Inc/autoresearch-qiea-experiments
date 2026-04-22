# autoresearch

This experiment uses a Quantum-Inspired Evolutionary Algorithm (QIEA) to choose binary architectural decisions for `train.py`.

## Core Objective

Minimize `val_bpb` under the fixed 5-minute training budget while searching over architecture choices encoded as q-bits.

## Files In Scope

- `train.py` - editable; this is where architecture is implemented.
- `qiea_optimizer.py` - produces architecture blueprints from q-bits.
- `qiea_evaluate.py` - consumes `val_bpb` and updates q-bits.
- `architecture_blueprint.txt` - plaintext blueprint for each run.
- `architecture_blueprint.json` - structured blueprint metadata.
- `results.tsv` - run log (untracked).

Do not modify `prepare.py`.

## One-Time Setup

1. Create a fresh experiment branch: `git checkout -b autoresearch/<tag>`.
2. Read `README.md`, `prepare.py`, and `train.py`.
3. Ensure dataset/tokenizer cache exists in `~/.cache/autoresearch/`; if missing run `uv run prepare.py`.
4. Initialize q-bit state once:
   - `uv run qiea_optimizer.py init --force`
5. Create `results.tsv` with header:
   - `commit\tval_bpb\tmemory_gb\tstatus\tbinary\tdescription`

## Mandatory QIEA Architecture Protocol (every run)

Follow this sequence exactly, every iteration:

1. Sample architecture q-bits:
   - `uv run qiea_optimizer.py sample`
2. Read `architecture_blueprint.txt`.
3. Open `train.py` and update only the constants in the block:
   - `# BEGIN QIEA_ARCH_BLUEPRINT`
   - `# END QIEA_ARCH_BLUEPRINT`
4. Apply the exact assignments from the blueprint (no deviations, no extra architecture tweaks).
5. Validate compilation before training:
   - `uv run python -m py_compile train.py`
6. Commit the architecture edit:
   - commit message must include the blueprint binary string (for traceability).
7. Run training (redirect output):
   - `uv run train.py > run.log 2>&1`
8. Confirm run success:
   - `grep "^val_bpb:\|^peak_vram_mb:" run.log`
   - if missing, inspect `tail -n 80 run.log`, fix only correctness bugs, and rerun.
9. Apply QIEA fitness update:
   - `uv run qiea_evaluate.py --run-log run.log`
10. Append one row to `results.tsv`:
   - commit short hash
   - `val_bpb`
   - memory in GB (`peak_vram_mb / 1024`)
   - status: `keep` / `discard` / `crash`
   - blueprint binary string
   - short architecture summary

## Keep/Discard Policy

- Lower `val_bpb` is better.
- Prefer simpler code when gains are marginal.
- If run improves best `val_bpb`, keep it.
- If run is worse/equal, mark discard and reset to previous best commit.

## Hard Constraints

- Do not change `prepare.py`.
- Do not install new packages.
- Ensure tensor shapes align and model compiles.
- Preserve the fixed 5-minute budget behavior.
- Do not skip the QIEA update step.

## QIEA Metrics To Preserve

The workflow must keep these artifacts up to date:

- `qiea_convergence.csv` (fitness progression)
- `qiea_trajectories.csv` (q-bit amplitudes)
- `qiea_architecture_frequency.csv` (choice frequencies)
- `qiea_diff_diversity.csv` (git diff diversity for `train.py`)

## Operating Mode

After setup, continue the loop autonomously until interrupted. Do not pause to ask whether to continue.
