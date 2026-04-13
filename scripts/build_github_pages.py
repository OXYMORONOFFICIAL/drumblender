#!/usr/bin/env python3
"""
Build a static GitHub Pages site for experiment summaries and audio audition.

The script discovers evaluation bundles under `logs/`, collects summary metrics,
and copies a small curated set of audio examples into `docs/` so the site can be
hosted as a fully static GitHub Pages deployment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_rel(path_str: str) -> str:
    return Path(path_str.replace("\\", "/")).as_posix()


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _slugify(text: str) -> str:
    out: list[str] = []
    prev_dash = False
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif ch in {"-", "_"} or ch.isspace():
            if not prev_dash:
                out.append("-")
                prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "item"


def _discover_result_dirs(logs_root: Path) -> list[Path]:
    found: dict[str, Path] = {}
    for summary_path in logs_root.rglob("summary.json"):
        if summary_path.parent.name != "evaluation":
            continue
        run_dir = summary_path.parent.parent
        found[str(run_dir.resolve())] = run_dir
    return sorted(found.values(), key=lambda p: str(p).lower())


def _merge_metric_rows(
    loss_rows: list[dict[str, str]] | None,
    metric_rows: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if not loss_rows and not metric_rows:
        return []
    if loss_rows is None:
        return metric_rows or []
    if metric_rows is None:
        return loss_rows

    key_candidates = ("index", "meta_key", "source_filename")
    chosen_key = None
    if loss_rows and metric_rows:
        for key in key_candidates:
            if key in loss_rows[0] and key in metric_rows[0]:
                chosen_key = key
                break

    if chosen_key is None:
        out: list[dict[str, str]] = []
        for loss_row, metric_row in zip(loss_rows, metric_rows):
            merged = dict(loss_row)
            merged.update(metric_row)
            out.append(merged)
        return out

    by_key: dict[str, dict[str, str]] = {}
    for row in loss_rows:
        key = str(row.get(chosen_key, "")).strip()
        if key:
            by_key[key] = row

    out = []
    for row in metric_rows:
        key = str(row.get(chosen_key, "")).strip()
        merged = {}
        if key and key in by_key:
            merged.update(by_key[key])
        merged.update(row)
        out.append(merged)
    return out


def _resolve_audio_path(run_dir: Path, raw_path: str, source_filename: str, role: str) -> Path | None:
    path = Path(raw_path) if raw_path else Path()
    if raw_path:
        if not path.is_absolute():
            path = run_dir / path
        if path.is_file():
            return path

    fallback = run_dir / role / Path(source_filename)
    if fallback.is_file():
        return fallback
    return None


def _pick_metric_key(rows: list[dict[str, Any]]) -> str | None:
    preferred = "test/loss"
    if any(preferred in row["metrics"] for row in rows):
        return preferred

    keys = sorted({key for row in rows for key in row["metrics"].keys()})
    return keys[0] if keys else None


def _pick_samples(rows: list[dict[str, Any]], sample_count: int, metric_key: str) -> list[dict[str, Any]]:
    if sample_count <= 0 or not rows:
        return []

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row["metrics"].get(metric_key, float("inf")),
            row["source_filename"],
        ),
    )

    n_best = max(1, sample_count // 3)
    n_mid = max(1, sample_count // 3)
    n_worst = max(1, sample_count - n_best - n_mid)

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_rows(candidates: list[dict[str, Any]], bucket: str, limit: int) -> None:
        bucket_count = 0
        for row in candidates:
            key = (row["source_filename"], bucket)
            if key in seen:
                continue
            enriched = dict(row)
            enriched["bucket"] = bucket
            selected.append(enriched)
            seen.add(key)
            bucket_count += 1
            if bucket_count >= limit:
                break

    add_rows(sorted_rows, "best", n_best)

    if n_mid > 0:
        center = len(sorted_rows) // 2
        radius = max(1, n_mid)
        start = max(0, center - radius)
        end = min(len(sorted_rows), center + radius + 1)
        add_rows(sorted_rows[start:end], "median", n_mid)

    if n_worst > 0:
        add_rows(list(reversed(sorted_rows)), "worst", n_worst)

    if len(selected) < sample_count:
        bucket_count = 0
        for row in sorted_rows:
            key = (row["source_filename"], "extra")
            if key in seen:
                continue
            enriched = dict(row)
            enriched["bucket"] = "extra"
            selected.append(enriched)
            seen.add(key)
            bucket_count += 1
            if len(selected) >= sample_count or bucket_count >= sample_count:
                break

    return selected[:sample_count]


def _copy_audio(src: Path, media_root: Path, run_slug: str, role: str, sample_id: int, source_filename: str) -> str:
    media_dir = media_root / run_slug / role
    media_dir.mkdir(parents=True, exist_ok=True)

    stem = _slugify(Path(source_filename).stem)[:48]
    digest = hashlib.sha1(source_filename.encode("utf-8")).hexdigest()[:8]
    dest = media_dir / f"{sample_id:02d}-{stem}-{digest}{src.suffix.lower()}"
    shutil.copy2(src, dest)
    return dest.relative_to(media_root.parent).as_posix()


def _build_audio_samples(
    run_dir: Path,
    run_slug: str,
    media_root: Path,
    sample_count: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    manifest_path = run_dir / "manifest.csv"
    loss_csv = run_dir / "evaluation" / "per_file_loss.csv"
    metric_csv = run_dir / "evaluation" / "per_file_metrics.csv"

    if not manifest_path.is_file() or not loss_csv.is_file():
        return [], []

    manifest_rows = _read_csv(manifest_path)
    loss_rows = _read_csv(loss_csv)
    metric_rows = _read_csv(metric_csv) if metric_csv.is_file() else None
    merged_metric_rows = _merge_metric_rows(loss_rows, metric_rows)

    metrics_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in merged_metric_rows:
        key = (
            str(row.get("index", "")).strip(),
            str(row.get("meta_key", "")).strip(),
            _safe_rel(str(row.get("source_filename", "")).strip()),
        )
        metrics_by_key[key] = row

    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        source_filename = _safe_rel(str(manifest_row.get("source_filename", "")).strip())
        key = (
            str(manifest_row.get("index", "")).strip(),
            str(manifest_row.get("meta_key", "")).strip(),
            source_filename,
        )

        merged = metrics_by_key.get(key, {})
        metrics: dict[str, float] = {}
        for metric_key, metric_value in merged.items():
            if metric_key in {"index", "meta_key", "source_filename", "length"}:
                continue
            numeric = _to_float(metric_value)
            if _is_finite(numeric):
                metrics[metric_key] = numeric

        recon = _resolve_audio_path(
            run_dir,
            str(manifest_row.get("recon_path", "")).strip(),
            source_filename,
            "recon",
        )
        target = _resolve_audio_path(
            run_dir,
            str(manifest_row.get("target_path", "")).strip(),
            source_filename,
            "target",
        )

        if recon is None:
            continue

        parts = Path(source_filename).parts
        rows.append(
            {
                "index": int(_to_float(manifest_row.get("index", -1))),
                "meta_key": str(manifest_row.get("meta_key", "")).strip(),
                "source_filename": source_filename,
                "length": int(_to_float(manifest_row.get("length", 0))),
                "pack": parts[0] if len(parts) > 1 else "",
                "recon_path": recon,
                "target_path": target,
                "metrics": metrics,
            }
        )

    metric_key = _pick_metric_key(rows)
    if metric_key is None:
        return [], []

    selected = _pick_samples(rows, sample_count=sample_count, metric_key=metric_key)

    output_samples: list[dict[str, Any]] = []
    metric_names = sorted({key for row in rows for key in row["metrics"].keys()})
    for sample_id, row in enumerate(selected, start=1):
        recon_rel = _copy_audio(
            row["recon_path"],
            media_root=media_root,
            run_slug=run_slug,
            role="recon",
            sample_id=sample_id,
            source_filename=row["source_filename"],
        )
        target_rel = None
        if row["target_path"] is not None:
            target_rel = _copy_audio(
                row["target_path"],
                media_root=media_root,
                run_slug=run_slug,
                role="target",
                sample_id=sample_id,
                source_filename=row["source_filename"],
            )

        output_samples.append(
            {
                "bucket": row["bucket"],
                "source_filename": row["source_filename"],
                "pack": row["pack"],
                "length": row["length"],
                "metric_focus": metric_key,
                "metrics": row["metrics"],
                "audio": {
                    "recon": recon_rel,
                    "target": target_rel,
                },
            }
        )

    return output_samples, metric_names


def _build_run_entry(run_dir: Path, media_root: Path, sample_count: int) -> dict[str, Any]:
    summary_path = run_dir / "evaluation" / "summary.json"
    summary = _read_json(summary_path)
    run_slug = _slugify(run_dir.name)

    samples, sample_metric_names = _build_audio_samples(
        run_dir=run_dir,
        run_slug=run_slug,
        media_root=media_root,
        sample_count=sample_count,
    )

    metrics = summary.get("metrics", {})
    loss_stats = summary.get("loss_stats", {})
    git_info = summary.get("git", {})

    return {
        "slug": run_slug,
        "name": run_dir.name,
        "display_name": run_dir.name.replace("_", " "),
        "relative_dir": run_dir.as_posix(),
        "config": summary.get("config"),
        "checkpoint": summary.get("ckpt"),
        "split": summary.get("split"),
        "num_items": summary.get("num_items"),
        "export_time_utc": summary.get("export_time_utc"),
        "metrics": metrics,
        "metric_names": sorted(metrics.keys()),
        "sample_metric_names": sample_metric_names,
        "loss_stats": loss_stats,
        "git_commit": git_info.get("commit"),
        "has_audio": bool(samples),
        "samples": samples,
    }


def _sort_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        runs,
        key=lambda run: (
            run["metrics"].get("test/loss", float("inf")),
            run["name"].lower(),
        ),
    )


def _write_site_data(output_dir: Path, payload: dict[str, Any]) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "site-data.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _clear_generated_dirs(output_dir: Path) -> None:
    for relative in ("data", "media"):
        target = output_dir / relative
        if target.exists():
            shutil.rmtree(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root directory to scan for evaluation outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs"),
        help="Destination directory for the static site.",
    )
    parser.add_argument(
        "--samples-per-run",
        type=int,
        default=6,
        help="How many audition samples to copy for each run with audio assets.",
    )
    parser.add_argument(
        "--site-title",
        type=str,
        default="DrumBlender Experiment Board",
        help="Title shown in the generated site data.",
    )
    args = parser.parse_args()

    logs_root = args.logs_root.resolve()
    output_dir = args.output.resolve()
    media_root = output_dir / "media"

    result_dirs = _discover_result_dirs(logs_root)
    if not result_dirs:
        raise SystemExit(f"No evaluation summaries found under: {logs_root}")

    _clear_generated_dirs(output_dir)

    runs = [
        _build_run_entry(run_dir, media_root=media_root, sample_count=args.samples_per_run)
        for run_dir in result_dirs
    ]
    runs = _sort_runs(runs)

    payload = {
        "site_title": args.site_title,
        "generated_from": logs_root.as_posix(),
        "generated_run_count": len(runs),
        "audio_run_count": sum(1 for run in runs if run["has_audio"]),
        "runs": runs,
    }
    _write_site_data(output_dir, payload)

    print(f"[github-pages] wrote site data for {len(runs)} runs -> {output_dir}")


if __name__ == "__main__":
    main()
