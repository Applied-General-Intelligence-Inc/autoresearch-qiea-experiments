"""
Run repeated QIEA architecture experiments (no baseline execution).

This script automates the run loop from EXPERIMENT.md for N runs:
1) sample q-bits -> blueprint
2) apply blueprint constants to train.py
3) compile check
4) run training
5) apply QIEA fitness update
6) append qiea_results.csv

Examples:
  uv run compare.py
  uv run compare.py --runs 100
  uv run compare.py --runs 100 --resume
  uv run compare.py --baseline-csv ../original/bench/original_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any


CSV_HEADER = [
    "run",
    "val_bpb",
    "peak_vram_mb",
    "training_seconds",
    "total_seconds",
    "status",
    "binary_string",
    "norm_type",
    "attention_type",
    "mlp_type",
    "value_embeddings",
    "qk_norm",
    "rope_base",
]

METRIC_KEYS = ["val_bpb", "peak_vram_mb", "training_seconds", "total_seconds"]


def _run_command(command: list[str], cwd: Path, log_path: Path | None = None) -> int:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        return int(completed.returncode)

    completed = subprocess.run(command, cwd=str(cwd), check=False, text=True)
    return int(completed.returncode)


def _extract_metrics(log_path: Path) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {k: None for k in METRIC_KEYS}
    if not log_path.exists():
        return metrics

    for line in log_path.read_text().splitlines():
        stripped = line.strip()
        for key in METRIC_KEYS:
            prefix = f"{key}:"
            if not stripped.startswith(prefix):
                continue
            value_str = stripped[len(prefix) :].strip().split()[0]
            try:
                metrics[key] = float(value_str)
            except ValueError:
                metrics[key] = None
    return metrics


def _write_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)


def _ensure_csv_with_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    _write_csv_header(path)


def _last_recorded_run(path: Path) -> int:
    if not path.exists():
        return 0

    last_run = 0
    with path.open() as f:
        for row in csv.DictReader(f):
            run_text = row.get("run")
            if run_text is None:
                continue
            try:
                run_id = int(float(run_text))
            except ValueError:
                continue
            last_run = max(last_run, run_id)
    return last_run


def _read_pending_qiea_run_id(qiea_repo: Path) -> int | None:
    state_path = qiea_repo / "qiea_state.json"
    if not state_path.exists():
        return None

    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return None

    pending = state.get("pending_observation")
    if not pending:
        return None

    run_id = pending.get("run_id")
    if run_id is None:
        return None
    try:
        return int(run_id)
    except (TypeError, ValueError):
        return None


def _load_blueprint_payload(qiea_repo: Path) -> dict[str, Any]:
    blueprint_json = qiea_repo / "architecture_blueprint.json"
    if blueprint_json.exists():
        return json.loads(blueprint_json.read_text())

    state_path = qiea_repo / "qiea_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        pending = state.get("pending_observation")
        if isinstance(pending, dict):
            return pending

    raise RuntimeError(
        "No blueprint payload found. Run qiea_optimizer.py sample first or resume from valid state."
    )


def _apply_blueprint_to_train(train_path: Path, assignments: dict[str, str]) -> None:
    lines = train_path.read_text().splitlines(keepends=True)

    begin_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if "# BEGIN QIEA_ARCH_BLUEPRINT" in line:
            begin_idx = idx
        if "# END QIEA_ARCH_BLUEPRINT" in line:
            end_idx = idx
            break

    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        raise RuntimeError("Could not find QIEA_ARCH_BLUEPRINT block in train.py")

    pattern = re.compile(r"^(\s*)([A-Z0-9_]+)\s*=\s*([^#\n]*)(\s*#.*)?$")
    seen: set[str] = set()

    for idx in range(begin_idx + 1, end_idx):
        line = lines[idx]
        match = pattern.match(line.rstrip("\n"))
        if not match:
            continue

        indent, key, _old_value, comment = match.groups()
        if key not in assignments:
            continue

        new_value = assignments[key]
        comment = comment or ""
        lines[idx] = f"{indent}{key} = {new_value}{comment}\n"
        seen.add(key)

    missing = sorted(set(assignments) - seen)
    if missing:
        raise RuntimeError(f"Missing constants in train.py QIEA block: {', '.join(missing)}")

    train_path.write_text("".join(lines))


def _append_csv_row(
    path: Path,
    run_idx: int,
    metrics: dict[str, float | None],
    blueprint: dict[str, Any],
) -> None:
    val = metrics["val_bpb"]
    status = "ok" if val is not None else "crash"

    choices = blueprint.get("choices", {})
    binary = blueprint.get("binary_string", "")

    if status == "ok":
        row = [
            run_idx,
            f"{metrics['val_bpb']:.8f}",
            f"{(metrics['peak_vram_mb'] or 0.0):.3f}",
            f"{(metrics['training_seconds'] or 0.0):.3f}",
            f"{(metrics['total_seconds'] or 0.0):.3f}",
            status,
            str(binary),
            str(choices.get("norm_type", "")),
            str(choices.get("attention_type", "")),
            str(choices.get("mlp_type", "")),
            str(choices.get("value_embeddings", "")),
            str(choices.get("qk_norm", "")),
            str(choices.get("rope_base", "")),
        ]
    else:
        row = [
            run_idx,
            "0.000000",
            "0.0",
            "0.0",
            "0.0",
            status,
            str(binary),
            str(choices.get("norm_type", "")),
            str(choices.get("attention_type", "")),
            str(choices.get("mlp_type", "")),
            str(choices.get("value_embeddings", "")),
            str(choices.get("qk_norm", "")),
            str(choices.get("rope_base", "")),
        ]

    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _load_ok_vals(path: Path) -> list[float]:
    if not path.exists():
        return []

    vals: list[float] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            status = (row.get("status") or "").strip().lower()
            val_text = (row.get("val_bpb") or "").strip()
            if not val_text:
                continue
            try:
                val = float(val_text)
            except ValueError:
                continue

            if status:
                if status in {"ok", "keep"}:
                    vals.append(val)
                continue

            if val > 0:
                vals.append(val)

    return vals


def _print_summary(qiea_csv: Path, baseline_csv: Path | None) -> None:
    qiea_vals = _load_ok_vals(qiea_csv)

    print("QIEA summary")
    print(f"  qiea_runs_ok:   {len(qiea_vals)}")
    if not qiea_vals:
        print("  Not enough successful QIEA runs to summarize.")
        return

    print(f"  qiea_best:      {min(qiea_vals):.6f}")
    print(f"  qiea_mean:      {statistics.mean(qiea_vals):.6f}")
    print(f"  qiea_median:    {statistics.median(qiea_vals):.6f}")

    if baseline_csv is None:
        return

    baseline_vals = _load_ok_vals(baseline_csv)
    if not baseline_vals:
        print(f"  baseline_csv:   {baseline_csv} (no parseable successful rows)")
        return

    baseline_best = min(baseline_vals)
    qiea_best = min(qiea_vals)
    print("Comparison vs baseline CSV")
    print(f"  baseline_csv:   {baseline_csv}")
    print(f"  baseline_ok:    {len(baseline_vals)}")
    print(f"  baseline_best:  {baseline_best:.6f}")
    print(f"  delta_best:     {baseline_best - qiea_best:+.6f} (positive favors QIEA)")


def _run_qiea_init(qiea_repo: Path) -> None:
    print("[qiea] initializing state with --force")
    rc = _run_command(["uv", "run", "qiea_optimizer.py", "init", "--force"], cwd=qiea_repo)
    if rc != 0:
        raise RuntimeError("Failed to initialize QIEA state")


def _run_qiea_once(
    run_idx: int,
    qiea_repo: Path,
    results_csv: Path,
    sample_first: bool,
    crash_val_bpb: float,
) -> None:
    train_py = qiea_repo / "train.py"
    train_log = qiea_repo / "bench" / f"qiea_run_{run_idx}.log"
    update_log = qiea_repo / "bench" / f"qiea_update_{run_idx}.log"

    sample_rc: int | str
    if sample_first:
        sample_rc = _run_command(["uv", "run", "qiea_optimizer.py", "sample"], cwd=qiea_repo)
        if sample_rc != 0:
            raise RuntimeError(f"Sampling failed for run {run_idx}")
    else:
        sample_rc = "skipped"

    blueprint = _load_blueprint_payload(qiea_repo)
    assignments = blueprint.get("constant_assignments")
    if not isinstance(assignments, dict):
        raise RuntimeError("Blueprint is missing constant_assignments")

    _apply_blueprint_to_train(train_py, assignments)

    compile_rc = _run_command(
        ["uv", "run", "python", "-m", "py_compile", "train.py"],
        cwd=qiea_repo,
    )
    if compile_rc != 0:
        raise RuntimeError(f"Compile check failed after applying blueprint on run {run_idx}")

    train_rc = _run_command(["uv", "run", "train.py"], cwd=qiea_repo, log_path=train_log)
    metrics = _extract_metrics(train_log)
    _append_csv_row(results_csv, run_idx, metrics, blueprint)

    if metrics["val_bpb"] is None:
        update_rc = _run_command(
            ["uv", "run", "qiea_evaluate.py", "--val-bpb", str(crash_val_bpb)],
            cwd=qiea_repo,
            log_path=update_log,
        )
    else:
        update_rc = _run_command(
            [
                "uv",
                "run",
                "qiea_evaluate.py",
                "--run-log",
                str(train_log.relative_to(qiea_repo)),
            ],
            cwd=qiea_repo,
            log_path=update_log,
        )

    status = "ok" if metrics["val_bpb"] is not None else "crash"
    print(
        f"[qiea] run={run_idx} status={status} val_bpb={metrics['val_bpb']} "
        f"sample_rc={sample_rc} compile_rc={compile_rc} train_rc={train_rc} update_rc={update_rc} "
        f"binary={blueprint.get('binary_string')}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run QIEA architecture benchmark loop (no baseline training)."
    )
    parser.add_argument("--runs", type=int, default=100, help="Number of QIEA runs (default: 100)")
    parser.add_argument(
        "--qiea-repo",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to experiment2 repo (default: directory containing this script)",
    )
    parser.add_argument(
        "--skip-qiea-init",
        action="store_true",
        help="Skip qiea_optimizer.py init --force at start",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing qiea_results.csv and pending state",
    )
    parser.add_argument(
        "--crash-val-bpb",
        type=float,
        default=5.0,
        help="Penalty val_bpb sent to qiea_evaluate on crashed runs (default: 5.0)",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help="Optional path to pre-collected baseline CSV for summary comparison",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be >= 1")

    qiea_repo = args.qiea_repo.expanduser().resolve()
    if not qiea_repo.exists() or not qiea_repo.is_dir():
        raise FileNotFoundError(f"QIEA repo path does not exist: {qiea_repo}")

    for name in ["train.py", "prepare.py", "pyproject.toml", "qiea_optimizer.py", "qiea_evaluate.py"]:
        if not (qiea_repo / name).exists():
            raise FileNotFoundError(f"Missing required file in {qiea_repo}: {name}")

    baseline_csv = args.baseline_csv.expanduser().resolve() if args.baseline_csv is not None else None

    qiea_results_csv = qiea_repo / "bench" / "qiea_results.csv"
    if args.resume:
        _ensure_csv_with_header(qiea_results_csv)
    else:
        _write_csv_header(qiea_results_csv)

    if args.resume and not args.skip_qiea_init:
        print("[resume] ignoring qiea init to preserve current state")
    elif not args.skip_qiea_init:
        _run_qiea_init(qiea_repo)

    qiea_start = _last_recorded_run(qiea_results_csv) + 1 if args.resume else 1
    pending_qiea_run = _read_pending_qiea_run_id(qiea_repo)
    run_pending_first = args.resume and pending_qiea_run == qiea_start

    print(
        "Starting QIEA benchmark "
        f"(runs={args.runs}, qiea_repo={qiea_repo}, resume={args.resume}, qiea_start={qiea_start})"
    )

    if args.resume and pending_qiea_run is not None:
        print(f"[resume] pending_observation detected for qiea run_id={pending_qiea_run}")

    if run_pending_first and qiea_start <= args.runs:
        print(f"[resume] completing pending qiea run={qiea_start} without re-sampling")
        _run_qiea_once(
            qiea_start,
            qiea_repo,
            qiea_results_csv,
            sample_first=False,
            crash_val_bpb=args.crash_val_bpb,
        )
        qiea_start += 1

    for run_idx in range(qiea_start, args.runs + 1):
        _run_qiea_once(
            run_idx,
            qiea_repo,
            qiea_results_csv,
            sample_first=True,
            crash_val_bpb=args.crash_val_bpb,
        )

    _print_summary(qiea_results_csv, baseline_csv)


if __name__ == "__main__":
    main()
