#!/usr/bin/env python3
"""
Print top/bottom sample extremes per metric from a result folder or CSV.

Supported inputs:
1) Result folder (e.g., extracted recon bundle):
   - <run_dir>/evaluation/per_file_loss.csv
   - optional: <run_dir>/evaluation/per_file_metrics.csv
2) Direct CSV path:
   - per_file_loss.csv
   - per_file_metrics.csv

If per_file_metrics.csv does not contain sample names, this script will try to
recover them from sibling per_file_loss.csv and merge by row order.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


# Ranking metric label -> canonical column name
RANK_METRICS = [
    ("MSS", "test/loss"),
    ("LSD", "test/lsd"),
    ("SF", "test/flux_onset"),
]

# Detail metrics to always print for selected samples
# NOTE: In this repository, MSS and Loss both map to test/loss.
DETAIL_METRICS = [
    ("MSS", "test/loss"),
    ("LSD", "test/lsd"),
    ("SF", "test/flux_onset"),
    ("Loss", "test/loss"),
]


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _resolve_paths(input_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Resolve per_file_loss.csv and per_file_metrics.csv paths from either:
    - result folder
    - evaluation folder
    - direct CSV path
    """
    if input_path.is_file():
        name = input_path.name.lower()
        if name == "per_file_loss.csv":
            loss_csv = input_path
            metrics_csv = input_path.parent / "per_file_metrics.csv"
            return loss_csv, metrics_csv if metrics_csv.is_file() else None
        if name == "per_file_metrics.csv":
            metrics_csv = input_path
            loss_csv = input_path.parent / "per_file_loss.csv"
            return (loss_csv if loss_csv.is_file() else None), metrics_csv
        return None, None

    # Directory input
    eval_dir = input_path / "evaluation"
    if eval_dir.is_dir():
        loss_csv = eval_dir / "per_file_loss.csv"
        metrics_csv = eval_dir / "per_file_metrics.csv"
        return (loss_csv if loss_csv.is_file() else None), (
            metrics_csv if metrics_csv.is_file() else None
        )

    # Maybe user passed evaluation dir directly
    loss_csv = input_path / "per_file_loss.csv"
    metrics_csv = input_path / "per_file_metrics.csv"
    return (loss_csv if loss_csv.is_file() else None), (
        metrics_csv if metrics_csv.is_file() else None
    )


def _merge_rows(
    loss_rows: Optional[List[Dict[str, str]]],
    metrics_rows: Optional[List[Dict[str, str]]],
) -> List[Dict[str, object]]:
    """
    Merge rows from loss/metrics CSVs.
    - If both exist and lengths match, merge by row order.
    - If only one exists, return that one.
    """
    if not loss_rows and not metrics_rows:
        return []

    merged: List[Dict[str, object]] = []

    if loss_rows is not None and metrics_rows is not None:
        # Prefer key-based merge when possible, fallback to row-order merge.
        # This keeps "source_filename" stable even when CSV row order differs.
        key_fields = ["index", "meta_key", "source_filename"]
        chosen_key = None
        for k in key_fields:
            if (
                len(loss_rows) > 0
                and len(metrics_rows) > 0
                and k in loss_rows[0]
                and k in metrics_rows[0]
            ):
                chosen_key = k
                break

        if chosen_key is not None:
            loss_by_key: Dict[str, Dict[str, str]] = {}
            for r in loss_rows:
                kv = str(r.get(chosen_key, "")).strip()
                if kv != "":
                    loss_by_key[kv] = r

            for i, mr in enumerate(metrics_rows):
                kv = str(mr.get(chosen_key, "")).strip()
                row: Dict[str, object] = {}
                if kv != "" and kv in loss_by_key:
                    row.update(loss_by_key[kv])
                row.update(mr)
                row["__row_idx__"] = i
                merged.append(row)
            return merged

        n = min(len(loss_rows), len(metrics_rows))
        for i in range(n):
            row = {}
            row.update(loss_rows[i])
            row.update(metrics_rows[i])
            row["__row_idx__"] = i
            merged.append(row)
        return merged

    if loss_rows is not None:
        for i, r in enumerate(loss_rows):
            row: Dict[str, object] = dict(r)
            row["__row_idx__"] = i
            merged.append(row)
        return merged

    # metrics_rows only
    assert metrics_rows is not None
    for i, r in enumerate(metrics_rows):
        row = dict(r)
        row["__row_idx__"] = i
        merged.append(row)
    return merged


def _sample_name(row: Dict[str, object]) -> str:
    for key in ("source_filename", "orig_relpath", "filename", "meta_key", "index"):
        if key in row and row[key] is not None and str(row[key]).strip() != "":
            return str(row[key])
    return f"row_{row.get('__row_idx__', 'unknown')}"


def _fmt_val(v: float) -> str:
    if not _is_finite(v):
        return "N/A"
    return f"{v:.6f}"


def _detail_line(row: Dict[str, object]) -> str:
    parts = []
    for label, col in DETAIL_METRICS:
        vv = _to_float(row.get(col))
        parts.append(f"{label}={_fmt_val(vv)}")
    return "  ".join(parts)


def _print_metric_extremes(rows: List[Dict[str, object]], label: str, col: str, top_k: int) -> None:
    values: List[Tuple[Dict[str, object], float]] = []
    for r in rows:
        if col not in r:
            continue
        v = _to_float(r[col])
        if _is_finite(v):
            values.append((r, v))

    if len(values) == 0:
        print(f"\n[{label}] column '{col}' not available.")
        return

    values_desc = sorted(values, key=lambda x: x[1], reverse=True)
    values_asc = sorted(values, key=lambda x: x[1])
    k = min(top_k, len(values))

    print(f"\n[{label}] column={col}  samples={len(values)}")
    print(f"Top {k} (largest values)")
    for i, (row, v) in enumerate(values_desc[:k], start=1):
        name = _sample_name(row)
        print(f"  {i:2d}. rank_value={v:.6f}  sample={name}")
        print(f"      {_detail_line(row)}")

    print(f"Bottom {k} (smallest values)")
    for i, (row, v) in enumerate(values_asc[:k], start=1):
        name = _sample_name(row)
        print(f"  {i:2d}. rank_value={v:.6f}  sample={name}")
        print(f"      {_detail_line(row)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=str,
        help="Result folder, evaluation folder, per_file_loss.csv, or per_file_metrics.csv",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many samples to show")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    loss_csv, metrics_csv = _resolve_paths(input_path)
    if loss_csv is None and metrics_csv is None:
        raise FileNotFoundError(
            "Could not find per_file_loss.csv or per_file_metrics.csv from input."
        )

    loss_rows = _read_csv(loss_csv) if loss_csv is not None else None
    metrics_rows = _read_csv(metrics_csv) if metrics_csv is not None else None
    rows = _merge_rows(loss_rows, metrics_rows)

    print("=== Input Summary ===")
    print(f"input_path:   {input_path}")
    print(f"loss_csv:     {loss_csv if loss_csv is not None else 'N/A'}")
    print(f"metrics_csv:  {metrics_csv if metrics_csv is not None else 'N/A'}")
    print(f"rows_loaded:  {len(rows)}")

    for label, col in RANK_METRICS:
        _print_metric_extremes(rows, label, col, top_k=max(1, int(args.top_k)))


if __name__ == "__main__":
    main()
