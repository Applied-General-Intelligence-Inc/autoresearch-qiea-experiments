"""
Generate progress-style plots from benchmark CSV files.

This script reads:
- original/bench/original_results.csv
- experiment2/bench/qiea_results.csv

and writes PNG plots in the same keep/discard + running-best style
used in progress.png.

Usage:
  uv run plot_compare.py
  uv run plot_compare.py --runs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_ok_results(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"run", "val_bpb", "status"}
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise RuntimeError(f"{label} CSV is missing columns: {missing_text}")

    df["run"] = pd.to_numeric(df["run"], errors="coerce")
    df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
    df["status"] = df["status"].astype(str).str.lower()

    ok = df[(df["status"] == "ok") & df["run"].notna() & df["val_bpb"].notna()].copy()
    if ok.empty:
        raise RuntimeError(f"{label} CSV has no successful runs (status=ok)")

    ok.sort_values("run", inplace=True)
    ok.reset_index(drop=True, inplace=True)
    ok["running_best"] = ok["val_bpb"].cummin()
    ok["is_keep"] = ok["val_bpb"].eq(ok["running_best"])
    return ok


def _plot_progress(df: pd.DataFrame, label: str, kept_color: str, out_path: Path) -> None:
    kept = df[df["is_keep"]]
    discarded = df[~df["is_keep"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(
        discarded["run"],
        discarded["val_bpb"],
        c="#cfcfcf",
        s=20,
        alpha=0.6,
        zorder=2,
        label="Discarded",
    )
    ax.scatter(
        kept["run"],
        kept["val_bpb"],
        c=kept_color,
        s=58,
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
        label="Kept",
    )
    ax.step(
        df["run"],
        df["running_best"],
        where="post",
        color=kept_color,
        linewidth=2.0,
        alpha=0.8,
        zorder=3,
        label="Running best",
    )

    n_total = len(df)
    n_kept = int(df["is_keep"].sum())
    ax.set_title(f"{label} Progress: {n_total} Runs, {n_kept} Kept Improvements", fontsize=14)
    ax.set_xlabel("Run #", fontsize=12)
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
    ax.grid(alpha=0.22)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(original: pd.DataFrame, qiea: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(
        original["run"],
        original["val_bpb"],
        c="#b0b0b0",
        s=20,
        alpha=0.45,
        zorder=1,
        label="Original runs",
    )
    ax.scatter(
        qiea["run"],
        qiea["val_bpb"],
        c="#85d8b2",
        s=22,
        alpha=0.45,
        zorder=1,
        label="Experiment2 QIEA runs",
    )

    ax.step(
        original["run"],
        original["running_best"],
        where="post",
        color="#34495e",
        linewidth=2.0,
        zorder=3,
        label="Original running best",
    )
    ax.step(
        qiea["run"],
        qiea["running_best"],
        where="post",
        color="#27ae60",
        linewidth=2.0,
        zorder=4,
        label="Experiment2 QIEA running best",
    )

    best_orig = original["running_best"].min()
    best_qiea = qiea["running_best"].min()
    delta = best_orig - best_qiea

    ax.set_title(
        "Experiment2 vs Original (Running Best)\n"
        f"original_best={best_orig:.6f}, experiment2_best={best_qiea:.6f}, "
        f"delta_best={delta:+.6f} (positive favors experiment2)",
        fontsize=13,
    )
    ax.set_xlabel("Run #", fontsize=12)
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
    ax.grid(alpha=0.22)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate progress-style benchmark plots from original and experiment2 QIEA CSV results."
    )
    parser.add_argument(
        "--original-csv",
        type=Path,
        default=here.parent / "original" / "bench" / "original_results.csv",
        help="Path to original benchmark CSV",
    )
    parser.add_argument(
        "--qiea-csv",
        type=Path,
        default=here / "bench" / "qiea_results.csv",
        help="Path to experiment2 QIEA benchmark CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here / "bench",
        help="Directory to write PNG outputs",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=0,
        help="If >0, only use the first N successful runs from each CSV",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    original = _load_ok_results(args.original_csv.expanduser().resolve(), "Original")
    qiea = _load_ok_results(args.qiea_csv.expanduser().resolve(), "Experiment2 QIEA")

    if args.runs > 0:
        original = original.head(args.runs).copy()
        qiea = qiea.head(args.runs).copy()
        if original.empty or qiea.empty:
            raise RuntimeError("Not enough successful rows after --runs filter")

        original["running_best"] = original["val_bpb"].cummin()
        original["is_keep"] = original["val_bpb"].eq(original["running_best"])
        qiea["running_best"] = qiea["val_bpb"].cummin()
        qiea["is_keep"] = qiea["val_bpb"].eq(qiea["running_best"])

    out_dir = args.out_dir.expanduser().resolve()

    out_original = out_dir / "progress_original.png"
    out_qiea = out_dir / "progress_qiea.png"
    out_compare = out_dir / "compare_running_best.png"

    _plot_progress(original, "Original", "#2d3e50", out_original)
    _plot_progress(qiea, "Experiment2 QIEA", "#27ae60", out_qiea)
    _plot_comparison(original, qiea, out_compare)

    print("Saved plots:")
    print(f"  {out_original}")
    print(f"  {out_qiea}")
    print(f"  {out_compare}")


if __name__ == "__main__":
    main()
