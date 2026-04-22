"""
Run and compare baseline AutoResearch vs QIEA AutoResearch.

This script replaces the shell snippets in EXPERIMENT.md Step 4 onward.

Example:
  uv run compare.py --original-repo ../autoresearch-original --runs 20
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path


CSV_HEADER = [
    "run",
    "val_bpb",
    "peak_vram_mb",
    "training_seconds",
    "total_seconds",
    "status",
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

    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        text=True,
    )
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


def _append_csv_row(path: Path, run_idx: int, metrics: dict[str, float | None]) -> None:
    val = metrics["val_bpb"]
    status = "ok" if val is not None else "crash"

    if status == "ok":
        row = [
            run_idx,
            f"{metrics['val_bpb']:.8f}",
            f"{(metrics['peak_vram_mb'] or 0.0):.3f}",
            f"{(metrics['training_seconds'] or 0.0):.3f}",
            f"{(metrics['total_seconds'] or 0.0):.3f}",
            status,
        ]
    else:
        row = [run_idx, "0.000000", "0.0", "0.0", "0.0", status]

    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _run_baseline_once(run_idx: int, original_repo: Path, results_csv: Path) -> None:
    run_log = original_repo / "bench" / f"original_run_{run_idx}.log"
    rc = _run_command(["uv", "run", "train.py"], cwd=original_repo, log_path=run_log)
    metrics = _extract_metrics(run_log)
    _append_csv_row(results_csv, run_idx, metrics)

    status = "ok" if metrics["val_bpb"] is not None else "crash"
    print(
        f"[baseline] run={run_idx} status={status} "
        f"val_bpb={metrics['val_bpb']} train_rc={rc}"
    )


def _run_qiea_init(qiea_repo: Path) -> None:
    print("[qiea] initializing state with --force")
    rc = _run_command(["uv", "run", "qiea_optimizer.py", "init", "--force"], cwd=qiea_repo)
    if rc != 0:
        raise RuntimeError("Failed to initialize QIEA state")


def _run_qiea_once(run_idx: int, qiea_repo: Path, results_csv: Path, sample_first: bool = True) -> None:
    train_log = qiea_repo / "bench" / f"qiea_run_{run_idx}.log"
    update_log = qiea_repo / "bench" / f"qiea_update_{run_idx}.log"
    train_log_arg = str(train_log.relative_to(qiea_repo))

    sample_rc: int | str
    if sample_first:
        sample_rc = _run_command(["uv", "run", "qiea_optimizer.py", "sample"], cwd=qiea_repo)
    else:
        sample_rc = "skipped"
    train_rc = _run_command(["uv", "run", "train.py"], cwd=qiea_repo, log_path=train_log)
    update_rc = _run_command(
        ["uv", "run", "qiea_evaluate.py", "--run-log", train_log_arg],
        cwd=qiea_repo,
        log_path=update_log,
    )

    metrics = _extract_metrics(train_log)
    _append_csv_row(results_csv, run_idx, metrics)

    status = "ok" if metrics["val_bpb"] is not None else "crash"
    print(
        f"[qiea] run={run_idx} status={status} val_bpb={metrics['val_bpb']} "
        f"sample_rc={sample_rc} train_rc={train_rc} update_rc={update_rc}"
    )


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


def _load_ok_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def _cum_best(values: list[float]) -> list[float]:
    out: list[float] = []
    best: float | None = None
    for value in values:
        best = value if best is None else min(best, value)
        out.append(best)
    return out


def _print_summary(original_csv: Path, qiea_csv: Path) -> None:
    orig_rows = _load_ok_rows(original_csv)
    qiea_rows = _load_ok_rows(qiea_csv)

    orig_vals = [float(r["val_bpb"]) for r in orig_rows]
    qiea_vals = [float(r["val_bpb"]) for r in qiea_rows]

    print("Comparison summary")
    print(f"  original_runs_ok: {len(orig_rows)}")
    print(f"  qiea_runs_ok:     {len(qiea_rows)}")

    if not orig_vals or not qiea_vals:
        print("  Not enough successful runs to compute full statistics.")
        return

    orig_best = min(orig_vals)
    qiea_best = min(qiea_vals)
    print(f"  original_best:    {orig_best:.6f}")
    print(f"  qiea_best:        {qiea_best:.6f}")
    print(f"  delta_best:       {orig_best - qiea_best:+.6f} (positive favors QIEA)")
    print(f"  original_mean:    {statistics.mean(orig_vals):.6f}")
    print(f"  qiea_mean:        {statistics.mean(qiea_vals):.6f}")
    print(f"  original_median:  {statistics.median(orig_vals):.6f}")
    print(f"  qiea_median:      {statistics.median(qiea_vals):.6f}")

    orig_curve = _cum_best(orig_vals)
    qiea_curve = _cum_best(qiea_vals)
    k = min(10, len(orig_curve), len(qiea_curve))
    if k > 0:
        print(f"  best_after_{k}_runs_original: {orig_curve[k - 1]:.6f}")
        print(f"  best_after_{k}_runs_qiea:     {qiea_curve[k - 1]:.6f}")


def _check_repo(path: Path, required_files: list[str]) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Repository path does not exist: {path}")
    for name in required_files:
        if not (path / name).exists():
            raise FileNotFoundError(f"Missing required file in {path}: {name}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline and QIEA benchmark loops, then print comparison summary."
    )
    parser.add_argument(
        "--original-repo",
        type=Path,
        required=True,
        help="Path to baseline karpathy/autoresearch checkout",
    )
    parser.add_argument(
        "--qiea-repo",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to QIEA repo (default: directory containing compare.py)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of runs for each variant (default: 20)",
    )
    parser.add_argument(
        "--protocol",
        choices=["sequential", "interleaved"],
        default="sequential",
        help="Run all baseline then all QIEA (sequential) or pair each iteration (interleaved)",
    )
    parser.add_argument(
        "--skip-qiea-init",
        action="store_true",
        help="Skip qiea_optimizer.py init --force before benchmark",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing CSV/state instead of starting from run 1",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be >= 1")

    original_repo = args.original_repo.expanduser().resolve()
    qiea_repo = args.qiea_repo.expanduser().resolve()

    _check_repo(original_repo, ["train.py", "prepare.py", "pyproject.toml"])
    _check_repo(
        qiea_repo,
        ["train.py", "prepare.py", "pyproject.toml", "qiea_optimizer.py", "qiea_evaluate.py"],
    )

    original_results_csv = original_repo / "bench" / "original_results.csv"
    qiea_results_csv = qiea_repo / "bench" / "qiea_results.csv"

    if args.resume:
        _ensure_csv_with_header(original_results_csv)
        _ensure_csv_with_header(qiea_results_csv)
    else:
        _write_csv_header(original_results_csv)
        _write_csv_header(qiea_results_csv)

    if args.resume and not args.skip_qiea_init:
        print("[resume] ignoring qiea init to preserve current state")
    elif not args.skip_qiea_init:
        _run_qiea_init(qiea_repo)

    original_start = 1
    qiea_start = 1
    if args.resume:
        original_start = _last_recorded_run(original_results_csv) + 1
        qiea_start = _last_recorded_run(qiea_results_csv) + 1

    pending_qiea_run = _read_pending_qiea_run_id(qiea_repo)
    run_pending_first = args.resume and pending_qiea_run == qiea_start

    print(
        "Starting benchmark "
        f"(runs={args.runs}, protocol={args.protocol}, original_repo={original_repo}, qiea_repo={qiea_repo}, "
        f"resume={args.resume}, original_start={original_start}, qiea_start={qiea_start})"
    )

    if args.resume and pending_qiea_run is not None:
        print(f"[resume] pending_observation detected for qiea run_id={pending_qiea_run}")

    if args.protocol == "sequential":
        for run_idx in range(original_start, args.runs + 1):
            _run_baseline_once(run_idx, original_repo, original_results_csv)

        if run_pending_first and qiea_start <= args.runs:
            print(f"[resume] completing pending qiea run={qiea_start} without re-sampling")
            _run_qiea_once(qiea_start, qiea_repo, qiea_results_csv, sample_first=False)
            qiea_start += 1

        for run_idx in range(qiea_start, args.runs + 1):
            _run_qiea_once(run_idx, qiea_repo, qiea_results_csv)
    else:
        for run_idx in range(1, args.runs + 1):
            if run_idx >= original_start:
                _run_baseline_once(run_idx, original_repo, original_results_csv)

            if run_pending_first and run_idx == qiea_start and run_idx <= args.runs:
                print(f"[resume] completing pending qiea run={qiea_start} without re-sampling")
                _run_qiea_once(run_idx, qiea_repo, qiea_results_csv, sample_first=False)
                run_pending_first = False
            elif run_idx >= qiea_start:
                _run_qiea_once(run_idx, qiea_repo, qiea_results_csv)

    _print_summary(original_results_csv, qiea_results_csv)


if __name__ == "__main__":
    main()