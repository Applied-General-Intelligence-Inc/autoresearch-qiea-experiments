"""
Quantum-Inspired Evolutionary Algorithm (QIEA) for binary architecture search.

This module manages architectural q-bits, collapses them into a binary blueprint,
and updates q-bit amplitudes using final val_bpb as fitness.

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
from typing import Any


STATE_PATH = Path("qiea_state.json")
BLUEPRINT_TEXT_PATH = Path("architecture_blueprint.txt")
BLUEPRINT_JSON_PATH = Path("architecture_blueprint.json")
CONVERGENCE_CSV_PATH = Path("qiea_convergence.csv")
TRAJECTORY_CSV_PATH = Path("qiea_trajectories.csv")
FREQUENCY_CSV_PATH = Path("qiea_architecture_frequency.csv")
DIVERSITY_CSV_PATH = Path("qiea_diff_diversity.csv")

MIN_ROTATION_ANGLE = 0.005
MAX_ROTATION_ANGLE = 0.12
NEGATIVE_FEEDBACK_SCALE = 0.5


@dataclass(frozen=True)
class ArchitectureQBitSpec:
    name: str
    zero_label: str
    one_label: str
    target_constant: str
    zero_value: str
    one_value: str
    rationale: str


ARCHITECTURE_QBITS = [
    ArchitectureQBitSpec(
        name="norm_type",
        zero_label="RMSNorm",
        one_label="LayerNorm",
        target_constant="ARCH_NORM_TYPE",
        zero_value='"rmsnorm"',
        one_value='"layernorm"',
        rationale="Controls token/activation normalization family.",
    ),
    ArchitectureQBitSpec(
        name="attention_type",
        zero_label="StandardAttention",
        one_label="SlidingWindowAttention",
        target_constant="ARCH_ATTENTION_TYPE",
        zero_value='"standard"',
        one_value='"sliding_window"',
        rationale="Global full context attention vs local sliding windows.",
    ),
    ArchitectureQBitSpec(
        name="mlp_type",
        zero_label="ReLU2MLP",
        one_label="SwiGLUMLP",
        target_constant="ARCH_MLP_TYPE",
        zero_value='"relu2"',
        one_value='"swiglu"',
        rationale="Feed-forward nonlinearity and projection structure.",
    ),
    ArchitectureQBitSpec(
        name="value_embeddings",
        zero_label="NoValueEmbeddings",
        one_label="UseValueEmbeddings",
        target_constant="ARCH_USE_VALUE_EMBEDDINGS",
        zero_value="False",
        one_value="True",
        rationale="Enables or disables residual value-embedding pathway.",
    ),
    ArchitectureQBitSpec(
        name="qk_norm",
        zero_label="NoQKNorm",
        one_label="UseQKNorm",
        target_constant="ARCH_USE_QK_NORM",
        zero_value="False",
        one_value="True",
        rationale="Applies normalization to q/k attention projections.",
    ),
    ArchitectureQBitSpec(
        name="rope_base",
        zero_label="RoPEBase10k",
        one_label="RoPEBase50k",
        target_constant="ARCH_ROPE_BASE",
        zero_value="10000.0",
        one_value="50000.0",
        rationale="Rotary frequency base, affects long-range phase behavior.",
    ),
]


def _initial_qbit() -> dict[str, float]:
    amp = 1.0 / math.sqrt(2.0)
    return {"alpha": amp, "beta": amp}


def _normalize_qbit(alpha: float, beta: float) -> tuple[float, float]:
    norm = math.sqrt(alpha * alpha + beta * beta)
    if norm == 0.0:
        init = _initial_qbit()
        return init["alpha"], init["beta"]
    return alpha / norm, beta / norm


def _ensure_csv_headers() -> None:
    if not CONVERGENCE_CSV_PATH.exists():
        header = [
            "run_id",
            "val_bpb",
            "improved",
            "best_val_bpb",
            "rotation_angle",
            "binary_string",
        ]
        header.extend(spec.name for spec in ARCHITECTURE_QBITS)
        with CONVERGENCE_CSV_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    if not TRAJECTORY_CSV_PATH.exists():
        with TRAJECTORY_CSV_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "val_bpb", "qbit_index", "name", "alpha", "beta", "p_one"])

    if not FREQUENCY_CSV_PATH.exists():
        header = ["run_id", "val_bpb", "binary_string"]
        header.extend(f"{spec.name}_choice" for spec in ARCHITECTURE_QBITS)
        header.extend(f"{spec.name}_one_frequency" for spec in ARCHITECTURE_QBITS)
        with FREQUENCY_CSV_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    if not DIVERSITY_CSV_PATH.exists():
        with DIVERSITY_CSV_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "run_id",
                    "val_bpb",
                    "binary_string",
                    "head_commit",
                    "base_commit",
                    "added_lines",
                    "deleted_lines",
                    "changed_lines",
                    "train_total_lines",
                    "change_ratio",
                    "diff_hash",
                    "method",
                    "error",
                ]
            )


def initialize_state(force: bool = False) -> dict[str, Any]:
    if STATE_PATH.exists() and not force:
        return load_state()

    state = {
        "version": 2,
        "run_count": 0,
        "best_val_bpb": None,
        "qbits": [_initial_qbit() for _ in ARCHITECTURE_QBITS],
        "choice_counts": {
            spec.name: {"0": 0, "1": 0}
            for spec in ARCHITECTURE_QBITS
        },
        "pending_observation": None,
        "last_result": None,
    }
    save_state(state)
    _ensure_csv_headers()
    return state


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return initialize_state(force=False)

    state = json.loads(STATE_PATH.read_text())

    qbits = state.get("qbits", [])
    if len(qbits) != len(ARCHITECTURE_QBITS):
        raise RuntimeError(
            "qiea_state.json has incompatible architecture q-bit count. "
            "Re-initialize with: uv run qiea_optimizer.py init --force"
        )

    if "choice_counts" not in state:
        state["choice_counts"] = {
            spec.name: {"0": 0, "1": 0}
            for spec in ARCHITECTURE_QBITS
        }
    else:
        for spec in ARCHITECTURE_QBITS:
            counts = state["choice_counts"].setdefault(spec.name, {"0": 0, "1": 0})
            counts.setdefault("0", 0)
            counts.setdefault("1", 0)

    _ensure_csv_headers()
    return state


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def _bit_to_choice(spec: ArchitectureQBitSpec, bit: int) -> str:
    return spec.one_label if bit == 1 else spec.zero_label


def _bit_to_assignment(spec: ArchitectureQBitSpec, bit: int) -> str:
    return spec.one_value if bit == 1 else spec.zero_value


def _render_blueprint_text(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("QIEA ARCHITECTURE BLUEPRINT")
    lines.append("===========================")
    lines.append(f"run_id: {payload['run_id']}")
    lines.append(f"binary_string: {payload['binary_string']}")
    lines.append("")
    lines.append("Semantics of each bit (left to right):")
    for idx, spec in enumerate(ARCHITECTURE_QBITS):
        lines.append(
            f"  bit[{idx}] {spec.name}: 0={spec.zero_label}, 1={spec.one_label}"
        )
    lines.append("")
    lines.append("Collapsed architecture for this run:")
    for spec in ARCHITECTURE_QBITS:
        choice = payload["choices"][spec.name]
        lines.append(f"  - {spec.name}: {choice}")
    lines.append("")
    lines.append("Mandatory train.py edits (exact constants):")
    for spec in ARCHITECTURE_QBITS:
        value = payload["constant_assignments"][spec.target_constant]
        lines.append(f"  {spec.target_constant} = {value}")
    lines.append("")
    lines.append("Execution checklist:")
    lines.append("  1) Open train.py and update only the QIEA architecture constant block.")
    lines.append("  2) Ensure the constants exactly match the lines above.")
    lines.append("  3) Run uv run train.py > run.log 2>&1")
    lines.append("  4) Run uv run qiea_evaluate.py --run-log run.log")
    lines.append("  5) Do not manually tweak architecture constants outside this blueprint.")
    return "\n".join(lines) + "\n"


def collapse_state(seed: int | None = None) -> dict[str, Any]:
    state = load_state()
    rng = random.Random(seed)

    bits: list[int] = []
    choices: dict[str, str] = {}
    assignments: dict[str, str] = {}

    for idx, spec in enumerate(ARCHITECTURE_QBITS):
        qbit = state["qbits"][idx]
        p_one = float(qbit["beta"]) ** 2
        bit = 1 if rng.random() < p_one else 0
        bits.append(bit)
        choices[spec.name] = _bit_to_choice(spec, bit)
        assignments[spec.target_constant] = _bit_to_assignment(spec, bit)

    binary_string = "".join(str(bit) for bit in bits)
    run_id = int(state["run_count"]) + 1

    if state.get("pending_observation") is not None:
        previous_id = state["pending_observation"].get("run_id")
        print(f"Warning: overwriting pending observation for run_id={previous_id}")

    payload = {
        "run_id": run_id,
        "bits": bits,
        "binary_string": binary_string,
        "choices": choices,
        "constant_assignments": assignments,
    }

    state["pending_observation"] = payload
    save_state(state)

    BLUEPRINT_JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    BLUEPRINT_TEXT_PATH.write_text(_render_blueprint_text(payload))

    print(
        f"Collapsed architecture q-bits for run_id={run_id} "
        f"-> {BLUEPRINT_TEXT_PATH} (binary={binary_string})"
    )
    return payload


def _rotation_angle(best_val_bpb: float | None, val_bpb: float, improved: bool) -> float:
    if best_val_bpb is None:
        base_angle = 0.5 * (MIN_ROTATION_ANGLE + MAX_ROTATION_ANGLE)
        return base_angle

    relative_delta = abs(best_val_bpb - val_bpb) / max(abs(best_val_bpb), 1e-12)
    scaled = min(1.0, relative_delta * 20.0)
    angle = MIN_ROTATION_ANGLE + (MAX_ROTATION_ANGLE - MIN_ROTATION_ANGLE) * scaled
    return angle if improved else angle * NEGATIVE_FEEDBACK_SCALE


def _update_single_qbit(qbit: dict[str, float], target_bit: int, angle: float) -> None:
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


def _append_convergence_row(
    run_id: int,
    val_bpb: float,
    improved: bool,
    best_val_bpb: float | None,
    rotation_angle: float,
    bits: list[int],
) -> None:
    row: list[str | int] = [
        run_id,
        f"{val_bpb:.8f}",
        int(improved),
        "" if best_val_bpb is None else f"{best_val_bpb:.8f}",
        f"{rotation_angle:.8f}",
        "".join(str(bit) for bit in bits),
    ]
    row.extend(bits)

    with CONVERGENCE_CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _append_trajectory_rows(state: dict[str, Any], run_id: int, val_bpb: float) -> None:
    with TRAJECTORY_CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        for idx, spec in enumerate(ARCHITECTURE_QBITS):
            qbit = state["qbits"][idx]
            p_one = float(qbit["beta"]) ** 2
            writer.writerow(
                [
                    run_id,
                    f"{val_bpb:.8f}",
                    idx,
                    spec.name,
                    f"{float(qbit['alpha']):.12f}",
                    f"{float(qbit['beta']):.12f}",
                    f"{p_one:.12f}",
                ]
            )


def _append_frequency_row(state: dict[str, Any], run_id: int, val_bpb: float, bits: list[int]) -> None:
    row: list[str | int] = [run_id, f"{val_bpb:.8f}", "".join(str(bit) for bit in bits)]

    for idx, spec in enumerate(ARCHITECTURE_QBITS):
        row.append(_bit_to_choice(spec, bits[idx]))

    total_runs = max(1, int(state["run_count"]))
    for spec in ARCHITECTURE_QBITS:
        one_count = int(state["choice_counts"][spec.name]["1"])
        row.append(f"{one_count / total_runs:.8f}")

    with FREQUENCY_CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _append_diversity_row(
    run_id: int,
    val_bpb: float,
    bits: list[int],
    diversity: dict[str, Any] | None,
) -> None:
    diversity = diversity or {}
    row = [
        run_id,
        f"{val_bpb:.8f}",
        "".join(str(bit) for bit in bits),
        diversity.get("head_commit", ""),
        diversity.get("base_commit", ""),
        diversity.get("added_lines", ""),
        diversity.get("deleted_lines", ""),
        diversity.get("changed_lines", ""),
        diversity.get("train_total_lines", ""),
        diversity.get("change_ratio", ""),
        diversity.get("diff_hash", ""),
        diversity.get("method", ""),
        diversity.get("error", ""),
    ]

    with DIVERSITY_CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def update_from_result(val_bpb: float, diversity: dict[str, Any] | None = None) -> dict[str, Any]:
    state = load_state()
    observation = state.get("pending_observation")
    if observation is None:
        raise RuntimeError(
            "No pending observation found. Run 'uv run qiea_optimizer.py sample' before updating."
        )

    run_id = int(observation["run_id"])
    bits = [int(bit) for bit in observation["bits"]]
    best_before = state.get("best_val_bpb")
    improved = best_before is None or val_bpb < float(best_before)
    angle = _rotation_angle(best_before, val_bpb, improved)

    for idx, sampled_bit in enumerate(bits):
        qbit = state["qbits"][idx]
        target_bit = sampled_bit if improved else 1 - sampled_bit
        _update_single_qbit(qbit, target_bit, angle)

    for idx, spec in enumerate(ARCHITECTURE_QBITS):
        bit = bits[idx]
        key = str(bit)
        state["choice_counts"][spec.name][key] = int(state["choice_counts"][spec.name][key]) + 1

    best_after = val_bpb if improved else best_before
    state["best_val_bpb"] = best_after
    state["run_count"] = max(int(state.get("run_count", 0)), run_id)
    state["last_result"] = {
        "run_id": run_id,
        "val_bpb": float(val_bpb),
        "improved": bool(improved),
        "rotation_angle": float(angle),
        "binary_string": observation["binary_string"],
        "diversity": diversity or {},
    }
    state["pending_observation"] = None

    _append_convergence_row(run_id, val_bpb, improved, best_after, angle, bits)
    _append_trajectory_rows(state, run_id, val_bpb)
    _append_frequency_row(state, run_id, val_bpb, bits)
    _append_diversity_row(run_id, val_bpb, bits, diversity)
    save_state(state)

    return {
        "run_id": run_id,
        "val_bpb": float(val_bpb),
        "improved": bool(improved),
        "best_val_bpb": None if best_after is None else float(best_after),
        "rotation_angle": float(angle),
        "binary_string": observation["binary_string"],
    }


def _mean_prob_one_for_qbit(qbit: dict[str, float]) -> float:
    return float(qbit["beta"]) ** 2


def print_status() -> None:
    state = load_state()
    print("QIEA architecture status")
    print(f"  state_file: {STATE_PATH}")
    print(f"  run_count: {state.get('run_count', 0)}")
    print(f"  best_val_bpb: {state.get('best_val_bpb')}")
    print(f"  pending_observation: {state.get('pending_observation') is not None}")

    print("  mean P(bit=1) per architectural q-bit:")
    for idx, spec in enumerate(ARCHITECTURE_QBITS):
        p_one = _mean_prob_one_for_qbit(state["qbits"][idx])
        print(f"    bit[{idx}] {spec.name:18s}: {p_one:.4f}")

    run_count = max(1, int(state.get("run_count", 0)))
    print("  empirical choice frequencies:")
    for spec in ARCHITECTURE_QBITS:
        one_count = int(state["choice_counts"][spec.name]["1"])
        zero_count = int(state["choice_counts"][spec.name]["0"])
        one_freq = one_count / run_count
        print(
            f"    {spec.name:18s}: "
            f"0={zero_count}, 1={one_count}, one_frequency={one_freq:.4f}"
        )

    print(f"  blueprint_txt: {BLUEPRINT_TEXT_PATH}")
    print(f"  blueprint_json: {BLUEPRINT_JSON_PATH}")
    print(f"  convergence_csv: {CONVERGENCE_CSV_PATH}")
    print(f"  trajectory_csv: {TRAJECTORY_CSV_PATH}")
    print(f"  frequency_csv: {FREQUENCY_CSV_PATH}")
    print(f"  diversity_csv: {DIVERSITY_CSV_PATH}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QIEA optimizer for architecture search")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Initialize architecture q-bit state")
    init_p.add_argument("--force", action="store_true", help="Overwrite existing state")

    sample_p = sub.add_parser("sample", help="Collapse q-bits to architecture blueprint")
    sample_p.add_argument("--seed", type=int, default=None, help="Optional random seed")

    update_p = sub.add_parser("update", help="Update q-bits from observed val_bpb")
    update_p.add_argument("--val-bpb", type=float, required=True, help="Observed final val_bpb")

    sub.add_parser("status", help="Show current q-bit summary")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "init":
        initialize_state(force=args.force)
        print(f"Initialized architecture q-bit state at {STATE_PATH}")
        return

    if args.command == "sample":
        payload = collapse_state(seed=args.seed)
        print(f"Sampled run_id={payload['run_id']} binary={payload['binary_string']}")
        return

    if args.command == "update":
        result = update_from_result(args.val_bpb)
        print(
            "Updated architecture q-bits "
            f"(run_id={result['run_id']}, binary={result['binary_string']}, "
            f"improved={result['improved']}, best_val_bpb={result['best_val_bpb']}, "
            f"rotation_angle={result['rotation_angle']:.6f})"
        )
        return

    if args.command == "status":
        print_status()
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
