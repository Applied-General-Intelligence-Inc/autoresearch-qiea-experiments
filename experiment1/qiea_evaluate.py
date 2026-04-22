"""
Post-training QIEA update script.

Reads final val_bpb from one of:
1) --val-bpb argument
2) qiea_last_result.json (written by train.py)
3) run.log summary output

Then applies QIEA rotation-gate updates and logs convergence/trajectory CSV rows.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from qiea_optimizer import update_from_result


VAL_BPB_PATTERN = re.compile(r"^val_bpb:\s*([0-9]*\.?[0-9]+)")


def _extract_val_bpb_from_result_file(path):
    payload = json.loads(path.read_text())
    if "val_bpb" not in payload:
        raise RuntimeError(f"Missing val_bpb in {path}")
    return float(payload["val_bpb"])


def _extract_val_bpb_from_log(path):
    val_bpb = None
    for line in path.read_text().splitlines():
        match = VAL_BPB_PATTERN.match(line.strip())
        if match:
            val_bpb = float(match.group(1))
    if val_bpb is None:
        raise RuntimeError(f"Could not find val_bpb in {path}")
    return val_bpb


def _resolve_val_bpb(args):
    if args.val_bpb is not None:
        return float(args.val_bpb)

    result_path = Path(args.result_file)
    if result_path.exists():
        return _extract_val_bpb_from_result_file(result_path)

    run_log_path = Path(args.run_log)
    if run_log_path.exists():
        return _extract_val_bpb_from_log(run_log_path)

    raise RuntimeError(
        "Could not resolve val_bpb. Provide --val-bpb or ensure qiea_last_result.json "
        "or run.log exists."
    )


def main():
    parser = argparse.ArgumentParser(description="Apply QIEA update after a training run")
    parser.add_argument("--val-bpb", type=float, default=None, help="Explicit final val_bpb")
    parser.add_argument(
        "--result-file",
        default="qiea_last_result.json",
        help="Path to result JSON written by train.py",
    )
    parser.add_argument("--run-log", default="run.log", help="Path to training run log")
    args = parser.parse_args()

    val_bpb = _resolve_val_bpb(args)
    result = update_from_result(val_bpb)
    print(f"val_bpb:       {val_bpb:.8f}")
    print(f"run_id:        {result['run_id']}")
    print(f"improved:      {result['improved']}")
    print(f"best_val_bpb:  {result['best_val_bpb']}")
    print(f"rotation_angle:{result['rotation_angle']:.6f}")


if __name__ == "__main__":
    main()
