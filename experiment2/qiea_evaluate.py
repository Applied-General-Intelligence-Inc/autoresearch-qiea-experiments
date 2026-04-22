"""
Post-training QIEA update for architecture search.

Resolves final val_bpb from CLI/result/log, computes commit-level diff diversity for
train.py, and applies rotation-gate updates through qiea_optimizer.update_from_result.

Usage:
  uv run qiea_evaluate.py --run-log run.log
  uv run qiea_evaluate.py --val-bpb 0.9979
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from qiea_optimizer import update_from_result


VAL_BPB_PATTERN = re.compile(r"^val_bpb:\s*([0-9]*\.?[0-9]+)")


def _extract_val_bpb_from_result_file(path: Path) -> float:
    payload = json.loads(path.read_text())
    if "val_bpb" not in payload:
        raise RuntimeError(f"Missing val_bpb in {path}")
    return float(payload["val_bpb"])


def _extract_val_bpb_from_log(path: Path) -> float:
    val_bpb = None
    for line in path.read_text().splitlines():
        match = VAL_BPB_PATTERN.match(line.strip())
        if match:
            val_bpb = float(match.group(1))
    if val_bpb is None:
        raise RuntimeError(f"Could not find val_bpb in {path}")
    return val_bpb


def _resolve_val_bpb(args: argparse.Namespace) -> float:
    if args.val_bpb is not None:
        return float(args.val_bpb)

    result_path = Path(args.result_file)
    if result_path.exists():
        return _extract_val_bpb_from_result_file(result_path)

    run_log_path = Path(args.run_log)
    if run_log_path.exists():
        return _extract_val_bpb_from_log(run_log_path)

    raise RuntimeError(
        "Could not resolve val_bpb. Provide --val-bpb or ensure result/log files exist."
    )


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _collect_commit_diff_diversity(train_path: Path) -> dict[str, Any]:
    base_result = {
        "head_commit": "",
        "base_commit": "",
        "added_lines": "",
        "deleted_lines": "",
        "changed_lines": "",
        "train_total_lines": "",
        "change_ratio": "",
        "diff_hash": "",
        "method": "unavailable",
        "error": "",
    }

    inside_repo = _run_git(["rev-parse", "--is-inside-work-tree"])
    if inside_repo.returncode != 0 or inside_repo.stdout.strip().lower() != "true":
        base_result["error"] = "not a git repository"
        return base_result

    head = _run_git(["rev-parse", "--short", "HEAD"])
    if head.returncode != 0:
        base_result["error"] = head.stderr.strip() or "failed to resolve HEAD"
        return base_result

    base = _run_git(["rev-parse", "--short", "HEAD~1"])
    if base.returncode != 0:
        base_result["head_commit"] = head.stdout.strip()
        base_result["error"] = "HEAD~1 unavailable (need at least 2 commits)"
        return base_result

    rel_train_path = str(train_path.as_posix())
    numstat = _run_git(["diff", "--numstat", "HEAD~1", "HEAD", "--", rel_train_path])
    if numstat.returncode != 0:
        base_result["head_commit"] = head.stdout.strip()
        base_result["base_commit"] = base.stdout.strip()
        base_result["error"] = numstat.stderr.strip() or "git diff --numstat failed"
        return base_result

    added_lines = 0
    deleted_lines = 0
    for line in numstat.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added_token, deleted_token, path_token = parts[0], parts[1], parts[2]
        if path_token != rel_train_path:
            continue
        if added_token.isdigit():
            added_lines = int(added_token)
        if deleted_token.isdigit():
            deleted_lines = int(deleted_token)

    patch = _run_git(["diff", "HEAD~1", "HEAD", "--", rel_train_path])
    diff_hash = ""
    if patch.returncode == 0:
        diff_hash = hashlib.sha256(patch.stdout.encode("utf-8")).hexdigest()[:16]

    train_total_lines = 0
    if train_path.exists():
        train_total_lines = len(train_path.read_text().splitlines())

    changed_lines = added_lines + deleted_lines
    change_ratio = changed_lines / max(1, train_total_lines)

    return {
        "head_commit": head.stdout.strip(),
        "base_commit": base.stdout.strip(),
        "added_lines": added_lines,
        "deleted_lines": deleted_lines,
        "changed_lines": changed_lines,
        "train_total_lines": train_total_lines,
        "change_ratio": f"{change_ratio:.8f}",
        "diff_hash": diff_hash,
        "method": "git_head_to_prev",
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply QIEA architecture update after a training run")
    parser.add_argument("--val-bpb", type=float, default=None, help="Explicit final val_bpb")
    parser.add_argument(
        "--result-file",
        default="qiea_last_result.json",
        help="Path to train.py result JSON (default: qiea_last_result.json)",
    )
    parser.add_argument("--run-log", default="run.log", help="Path to run log (default: run.log)")
    parser.add_argument(
        "--train-file",
        default="train.py",
        help="Path to train.py for diff-diversity metrics",
    )
    args = parser.parse_args()

    val_bpb = _resolve_val_bpb(args)
    diversity = _collect_commit_diff_diversity(Path(args.train_file))
    result = update_from_result(val_bpb, diversity=diversity)

    print(f"val_bpb:            {val_bpb:.8f}")
    print(f"run_id:             {result['run_id']}")
    print(f"binary_string:      {result['binary_string']}")
    print(f"improved:           {result['improved']}")
    print(f"best_val_bpb:       {result['best_val_bpb']}")
    print(f"rotation_angle:     {result['rotation_angle']:.6f}")
    print(f"diff_method:        {diversity.get('method', '')}")
    print(f"diff_changed_lines: {diversity.get('changed_lines', '')}")
    print(f"diff_change_ratio:  {diversity.get('change_ratio', '')}")
    if diversity.get("error"):
        print(f"diff_error:         {diversity['error']}")


if __name__ == "__main__":
    main()
