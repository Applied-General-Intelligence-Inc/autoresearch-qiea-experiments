"""
Quantum-Inspired Evolutionary Algorithm (QIEA) for continuous hyperparameter search.

This module manages a population of q-bits (alpha, beta amplitudes), collapses them
into classical hyperparameter values, and applies rotation-gate updates based on the
observed val_bpb metric.

Usage:
  uv run qiea_optimizer.py init
  uv run qiea_optimizer.py sample
  uv run qiea_optimizer.py update --val-bpb 0.9979
  uv run qiea_optimizer.py status
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


STATE_PATH = Path("qiea_state.json")
HYPERPARAMS_PATH = Path("hyperparams.json")
TRAJECTORY_CSV_PATH = Path("qiea_trajectories.csv")
CONVERGENCE_CSV_PATH = Path("qiea_convergence.csv")

NUM_QBITS_PER_PARAM = 12
MIN_ROTATION_ANGLE = 0.005
MAX_ROTATION_ANGLE = 0.12
NEGATIVE_FEEDBACK_SCALE = 0.5


@dataclass(frozen=True)
class HyperparamSpec:
    min_value: float
    max_value: float


HYPERPARAM_SPECS = {
    "embedding_lr": HyperparamSpec(0.05, 1.20),
    "unembedding_lr": HyperparamSpec(0.0005, 0.02),
    "matrix_lr": HyperparamSpec(0.005, 0.08),
    "scalar_lr": HyperparamSpec(0.05, 1.00),
    "weight_decay": HyperparamSpec(0.0, 0.5),
    "adam_beta1": HyperparamSpec(0.6, 0.98),
    "adam_beta2": HyperparamSpec(0.85, 0.999),
    "warmup_ratio": HyperparamSpec(0.0, 0.30),
    "warmdown_ratio": HyperparamSpec(0.1, 0.90),
    "dropout": HyperparamSpec(0.0, 0.40),
}


def _initial_qbit():
    amp = 1.0 / math.sqrt(2.0)
    return {"alpha": amp, "beta": amp}


def _normalize_qbit(alpha, beta):
    norm = math.sqrt(alpha * alpha + beta * beta)
    if norm == 0.0:
        return _initial_qbit()["alpha"], _initial_qbit()["beta"]
    return alpha / norm, beta / norm


def _decode_bits_to_float(bits, spec):
    max_int = (1 << len(bits)) - 1
    bit_value = int("".join(str(bit) for bit in bits), 2)
    fraction = bit_value / max_int
    return spec.min_value + fraction * (spec.max_value - spec.min_value)


def _ensure_csv_headers():
    if not CONVERGENCE_CSV_PATH.exists():
        header = ["run_id", "val_bpb", "improved", "best_val_bpb"] + list(HYPERPARAM_SPECS.keys())
        with CONVERGENCE_CSV_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    if not TRAJECTORY_CSV_PATH.exists():
        with TRAJECTORY_CSV_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "val_bpb", "param", "qbit_index", "alpha", "beta"])


def initialize_state(force=False):
    if STATE_PATH.exists() and not force:
        return load_state()

    state = {
        "version": 1,
        "num_qbits_per_param": NUM_QBITS_PER_PARAM,
        "run_count": 0,
        "best_val_bpb": None,
        "params": {},
        "pending_observation": None,
        "last_result": None,
    }

    for name, spec in HYPERPARAM_SPECS.items():
        state["params"][name] = {
            "range": {"min": spec.min_value, "max": spec.max_value},
            "qbits": [_initial_qbit() for _ in range(NUM_QBITS_PER_PARAM)],
        }

    save_state(state)
    _ensure_csv_headers()
    return state


def load_state():
    if not STATE_PATH.exists():
        return initialize_state(force=False)

    state = json.loads(STATE_PATH.read_text())

    if state.get("num_qbits_per_param") != NUM_QBITS_PER_PARAM:
        raise RuntimeError(
            "qiea_state.json has a different num_qbits_per_param. "
            "Re-initialize with: uv run qiea_optimizer.py init --force"
        )

    missing = [name for name in HYPERPARAM_SPECS if name not in state.get("params", {})]
    if missing:
        raise RuntimeError(
            "qiea_state.json is missing parameters: "
            f"{', '.join(missing)}. Re-initialize with: uv run qiea_optimizer.py init --force"
        )

    _ensure_csv_headers()
    return state


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def collapse_state(seed=None):
    state = load_state()
    rng = random.Random(seed)

    collapsed = {}
    bitstrings = {}
    for name, spec in HYPERPARAM_SPECS.items():
        qbits = state["params"][name]["qbits"]
        bits = []
        for qbit in qbits:
            p_one = float(qbit["beta"]) ** 2
            bits.append(1 if rng.random() < p_one else 0)
        collapsed[name] = _decode_bits_to_float(bits, spec)
        bitstrings[name] = "".join(str(bit) for bit in bits)

    run_id = int(state["run_count"]) + 1
    if state.get("pending_observation") is not None:
        old_run_id = state["pending_observation"].get("run_id")
        print(f"Warning: overwriting pending observation for run_id={old_run_id}")

    observation = {
        "run_id": run_id,
        "collapsed": collapsed,
        "bitstrings": bitstrings,
    }
    state["pending_observation"] = observation
    save_state(state)

    HYPERPARAMS_PATH.write_text(json.dumps(collapsed, indent=2, sort_keys=True))
    print(f"Collapsed q-bits for run_id={run_id} -> {HYPERPARAMS_PATH}")
    return observation


def _rotation_angle(best_val_bpb, val_bpb, improved):
    if best_val_bpb is None:
        return 0.5 * (MIN_ROTATION_ANGLE + MAX_ROTATION_ANGLE)

    relative_delta = abs(best_val_bpb - val_bpb) / max(abs(best_val_bpb), 1e-12)
    scaled = min(1.0, relative_delta * 20.0)
    angle = MIN_ROTATION_ANGLE + (MAX_ROTATION_ANGLE - MIN_ROTATION_ANGLE) * scaled
    return angle if improved else angle * NEGATIVE_FEEDBACK_SCALE


def _update_single_qbit(qbit, target_bit, angle):
    alpha = float(qbit["alpha"])
    beta = float(qbit["beta"])
    alpha, beta = _normalize_qbit(alpha, beta)

    phi = math.atan2(beta, alpha)
    phi = min(max(phi, 0.0), math.pi / 2.0)
    target_phi = math.pi / 2.0 if target_bit == 1 else 0.0

    if abs(target_phi - phi) <= angle:
        phi_new = target_phi
    else:
        phi_new = phi + angle if target_phi > phi else phi - angle

    phi_new = min(max(phi_new, 0.0), math.pi / 2.0)
    alpha_new, beta_new = _normalize_qbit(math.cos(phi_new), math.sin(phi_new))
    qbit["alpha"] = alpha_new
    qbit["beta"] = beta_new


def _append_convergence_row(run_id, val_bpb, improved, best_val_bpb, collapsed):
    row = [
        run_id,
        f"{val_bpb:.8f}",
        int(improved),
        "" if best_val_bpb is None else f"{best_val_bpb:.8f}",
    ]
    row.extend(f"{collapsed[name]:.10f}" for name in HYPERPARAM_SPECS)
    with CONVERGENCE_CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _append_trajectory_rows(state, run_id, val_bpb):
    with TRAJECTORY_CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        for name in HYPERPARAM_SPECS:
            qbits = state["params"][name]["qbits"]
            for idx, qbit in enumerate(qbits):
                writer.writerow(
                    [
                        run_id,
                        f"{val_bpb:.8f}",
                        name,
                        idx,
                        f"{float(qbit['alpha']):.12f}",
                        f"{float(qbit['beta']):.12f}",
                    ]
                )


def update_from_result(val_bpb):
    state = load_state()
    observation = state.get("pending_observation")
    if observation is None:
        raise RuntimeError(
            "No pending observation found. Run 'uv run qiea_optimizer.py sample' before updating."
        )

    run_id = int(observation["run_id"])
    collapsed = observation["collapsed"]
    bitstrings = observation["bitstrings"]

    best_before = state.get("best_val_bpb")
    improved = best_before is None or val_bpb < float(best_before)
    angle = _rotation_angle(best_before, val_bpb, improved)

    for name in HYPERPARAM_SPECS:
        bits = [int(ch) for ch in bitstrings[name]]
        qbits = state["params"][name]["qbits"]
        for idx, qbit in enumerate(qbits):
            sampled_bit = bits[idx]
            target_bit = sampled_bit if improved else 1 - sampled_bit
            _update_single_qbit(qbit, target_bit, angle)

    best_after = val_bpb if improved else best_before
    state["best_val_bpb"] = best_after
    state["run_count"] = max(int(state.get("run_count", 0)), run_id)
    state["last_result"] = {
        "run_id": run_id,
        "val_bpb": float(val_bpb),
        "improved": bool(improved),
        "rotation_angle": float(angle),
    }
    state["pending_observation"] = None

    _append_convergence_row(run_id, val_bpb, improved, best_after, collapsed)
    _append_trajectory_rows(state, run_id, val_bpb)
    save_state(state)

    return {
        "run_id": run_id,
        "val_bpb": float(val_bpb),
        "improved": bool(improved),
        "best_val_bpb": None if best_after is None else float(best_after),
        "rotation_angle": float(angle),
    }


def _mean_prob_one_for_param(state, name):
    probs = []
    for qbit in state["params"][name]["qbits"]:
        probs.append(float(qbit["beta"]) ** 2)
    return sum(probs) / len(probs)


def print_status():
    state = load_state()
    print("QIEA status")
    print(f"  state_file: {STATE_PATH}")
    print(f"  run_count: {state.get('run_count', 0)}")
    print(f"  best_val_bpb: {state.get('best_val_bpb')}")
    print(f"  pending_observation: {state.get('pending_observation') is not None}")
    print("  mean P(bit=1) per hyperparameter:")
    for name in HYPERPARAM_SPECS:
        print(f"    {name:16s}: {_mean_prob_one_for_param(state, name):.4f}")
    print(f"  convergence_csv: {CONVERGENCE_CSV_PATH}")
    print(f"  trajectory_csv: {TRAJECTORY_CSV_PATH}")


def _build_parser():
    parser = argparse.ArgumentParser(description="QIEA optimizer for AutoResearch")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Initialize q-bit population state")
    init_p.add_argument("--force", action="store_true", help="Overwrite existing state")

    sample_p = sub.add_parser("sample", help="Collapse q-bits to hyperparams.json")
    sample_p.add_argument("--seed", type=int, default=None, help="Optional random seed")

    update_p = sub.add_parser("update", help="Update q-bits from a val_bpb result")
    update_p.add_argument("--val-bpb", type=float, required=True, help="Observed final val_bpb")

    sub.add_parser("status", help="Show state summary")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "init":
        initialize_state(force=args.force)
        print(f"Initialized state at {STATE_PATH}")
        return

    if args.command == "sample":
        obs = collapse_state(seed=args.seed)
        print(f"Sampled run_id={obs['run_id']}")
        return

    if args.command == "update":
        result = update_from_result(args.val_bpb)
        print(
            "Updated q-bits "
            f"(run_id={result['run_id']}, improved={result['improved']}, "
            f"best_val_bpb={result['best_val_bpb']}, rotation_angle={result['rotation_angle']:.6f})"
        )
        return

    if args.command == "status":
        print_status()
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
