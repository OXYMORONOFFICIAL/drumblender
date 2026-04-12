#!/usr/bin/env python3
"""
Postprocess exported reconstruction bundles.

This script unifies the previous summary-table and metric-extremes workflows:
- scan one run directory (or one bundle directory)
- ensure per-file metric cache exists
- write run-level summary tables
- write per-bundle top/bottom sample tables with fixed k=3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

try:
    import torchaudio  # type: ignore
except Exception:
    torchaudio = None

try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None


TOP_K = 3
RANK_METRICS = [
    ("MSS", "test/loss"),
    ("LSD", "test/lsd"),
    ("SF", "test/flux_onset"),
]
SUMMARY_HEADERS = [
    "bundle",
    "pack",
    "items",
    "mss_mean",
    "mss_std",
    "lsd_mean",
    "lsd_std",
    "sf_mean",
    "sf_std",
]


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    xs = [x for x in values if _is_finite(x)]
    if len(xs) == 0:
        return float("nan"), float("nan")
    mean = float(statistics.fmean(xs))
    std = float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0
    return mean, std


def _fmt_float(value: float, digits: int = 6) -> str:
    if not _is_finite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _escape_cell(value: object) -> str:
    text = str(value)
    return text.replace("|", r"\|")


def _markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    header_line = "| " + " | ".join(_escape_cell(h) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [
        "| " + " | ".join(_escape_cell(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + body_lines)


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def _bundle_dirs(root: Path) -> List[Path]:
    direct_summary = root / "evaluation" / "summary.json"
    if direct_summary.is_file():
        return [root]

    bundles = sorted(
        {
            summary_path.parents[1]
            for summary_path in root.rglob("summary.json")
            if summary_path.is_file() and summary_path.parent.name == "evaluation"
        },
        key=lambda p: p.as_posix(),
    )
    if len(bundles) == 0:
        raise RuntimeError(f"No evaluation bundles found under: {root}")
    return bundles


def _bundle_relpath(root: Path, bundle_dir: Path) -> str:
    try:
        rel = bundle_dir.relative_to(root)
        return "." if str(rel) == "." else rel.as_posix()
    except Exception:
        return bundle_dir.name


def _safe_report_name(relpath: str) -> str:
    if relpath in ("", "."):
        return "root"
    return relpath.replace("/", "__").replace("\\", "__")


def _load_audio_mono(path: Path) -> Tuple[torch.Tensor, int]:
    if torchaudio is not None:
        try:
            waveform, sr = torchaudio.load(str(path))
            return waveform[:1, :].to(torch.float32), int(sr)
        except Exception:
            pass

    if sf is not None:
        try:
            data, sr = sf.read(str(path), always_2d=True)
            mono = torch.from_numpy(data[:, :1].T).to(torch.float32)
            return mono, int(sr)
        except Exception:
            pass

    raise RuntimeError(f"Failed to read audio: {path}")


def _resample_mono(waveform: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return waveform

    if torchaudio is not None:
        try:
            return torchaudio.functional.resample(waveform, sr_in, sr_out)
        except Exception:
            pass

    n_in = int(waveform.shape[-1])
    n_out = max(1, int(round(n_in * float(sr_out) / float(sr_in))))
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0), size=n_out, mode="linear", align_corners=False
    ).squeeze(0)


def _pad_to_min_length(x: torch.Tensor, min_len: int) -> torch.Tensor:
    if x.shape[-1] >= min_len:
        return x
    return torch.nn.functional.pad(x, (0, int(min_len - x.shape[-1])))


def _lsd_single(
    x: torch.Tensor, y: torch.Tensor, n_fft: int = 8092, hop: int = 64
) -> float:
    eps = 1e-8
    min_len = max(int(n_fft), int(n_fft // 2 + 1))
    x = _pad_to_min_length(x, min_len)
    y = _pad_to_min_length(y, min_len)
    win = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(
        x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True, pad_mode="constant"
    )
    Y = torch.stft(
        y, n_fft=n_fft, hop_length=hop, window=win, return_complex=True, pad_mode="constant"
    )
    X = torch.log(torch.square(torch.abs(X)) + eps)
    Y = torch.log(torch.square(torch.abs(Y)) + eps)
    lsd = torch.mean(torch.square(X - Y), dim=-2)
    lsd = torch.mean(torch.sqrt(lsd), dim=-1)
    return float(lsd.squeeze(0).cpu())


def _sf_single(
    x: torch.Tensor, y: torch.Tensor, n_fft: int = 1024, hop: int = 64
) -> float:
    min_len = max(int(n_fft), int(n_fft // 2 + 1))
    x = _pad_to_min_length(x, min_len)
    y = _pad_to_min_length(y, min_len)
    win = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        window=win,
        return_complex=True,
        pad_mode="constant",
        normalized=False,
        onesided=True,
    )
    Y = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop,
        window=win,
        return_complex=True,
        pad_mode="constant",
        normalized=False,
        onesided=True,
    )
    flux_x = torch.diff(torch.abs(X), dim=1)
    flux_y = torch.diff(torch.abs(Y), dim=1)
    flux_x = (flux_x + torch.abs(flux_x)) / 2
    flux_y = (flux_y + torch.abs(flux_y)) / 2
    flux_x = torch.square(flux_x).sum(dim=1)
    flux_y = torch.square(flux_y).sum(dim=1)
    onset_err = torch.mean(torch.abs(flux_x - flux_y), dim=-1)
    return float(onset_err.squeeze(0).cpu())


def _sample_name(row: Dict[str, object]) -> str:
    for key in ("source_filename", "orig_relpath", "filename", "meta_key", "index"):
        if key in row and row[key] is not None and str(row[key]).strip() != "":
            return str(row[key])
    return f"row_{row.get('__row_idx__', 'unknown')}"


def _merge_rows(
    loss_rows: Optional[List[Dict[str, str]]],
    metrics_rows: Optional[List[Dict[str, str]]],
) -> List[Dict[str, object]]:
    if not loss_rows and not metrics_rows:
        return []

    merged: List[Dict[str, object]] = []
    if loss_rows is not None and metrics_rows is not None:
        key_fields = ["index", "meta_key", "source_filename"]
        chosen_key = None
        for key in key_fields:
            if (
                len(loss_rows) > 0
                and len(metrics_rows) > 0
                and key in loss_rows[0]
                and key in metrics_rows[0]
            ):
                chosen_key = key
                break

        if chosen_key is not None:
            loss_by_key: Dict[str, Dict[str, str]] = {}
            for row in loss_rows:
                key_value = str(row.get(chosen_key, "")).strip()
                if key_value != "":
                    loss_by_key[key_value] = row

            for i, metric_row in enumerate(metrics_rows):
                key_value = str(metric_row.get(chosen_key, "")).strip()
                row: Dict[str, object] = {}
                if key_value != "" and key_value in loss_by_key:
                    row.update(loss_by_key[key_value])
                row.update(metric_row)
                row["__row_idx__"] = i
                merged.append(row)
            return merged

        n = min(len(loss_rows), len(metrics_rows))
        for i in range(n):
            row: Dict[str, object] = {}
            row.update(loss_rows[i])
            row.update(metrics_rows[i])
            row["__row_idx__"] = i
            merged.append(row)
        return merged

    if loss_rows is not None:
        for i, row in enumerate(loss_rows):
            merged_row: Dict[str, object] = dict(row)
            merged_row["__row_idx__"] = i
            merged.append(merged_row)
        return merged

    assert metrics_rows is not None
    for i, row in enumerate(metrics_rows):
        merged_row = dict(row)
        merged_row["__row_idx__"] = i
        merged.append(merged_row)
    return merged


def _metrics_cache_complete(rows: List[Dict[str, object]], expected_len: int) -> bool:
    if len(rows) != expected_len or len(rows) == 0:
        return False
    required = {"test/loss", "test/lsd", "test/flux_onset"}
    for row in rows:
        if not required.issubset(row.keys()):
            return False
    return True


def _compute_metric_rows(bundle_dir: Path, loss_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    recon_root = bundle_dir / "recon"
    target_root = bundle_dir / "target"
    if not recon_root.exists() or not target_root.exists():
        raise FileNotFoundError(
            f"Missing recon/target directories under {bundle_dir}. "
            "Re-run export with SAVE_TARGET=on."
        )

    metric_rows: List[Dict[str, object]] = []
    for i, loss_row in enumerate(loss_rows):
        src_rel = str(loss_row.get("source_filename", "")).replace("\\", "/")
        rel_path = Path(src_rel)
        recon_path = recon_root / rel_path
        target_path = target_root / rel_path

        lsd = float("nan")
        sf_value = float("nan")
        if recon_path.is_file() and target_path.is_file():
            recon, sr_r = _load_audio_mono(recon_path)
            target, sr_t = _load_audio_mono(target_path)
            if sr_r != sr_t:
                recon = _resample_mono(recon, sr_r, sr_t)
            lsd = _lsd_single(recon, target)
            sf_value = _sf_single(recon, target)

        row: Dict[str, object] = {
            "index": loss_row.get("index", i),
            "meta_key": loss_row.get("meta_key", ""),
            "source_filename": loss_row.get("source_filename", ""),
            "length": loss_row.get("length", ""),
            "test/loss": _to_float(loss_row.get("test/loss")),
            "test/lsd": lsd,
            "test/flux_onset": sf_value,
        }
        metric_rows.append(row)

    cache_path = bundle_dir / "evaluation" / "per_file_metrics.csv"
    headers = [
        "index",
        "meta_key",
        "source_filename",
        "length",
        "test/loss",
        "test/lsd",
        "test/flux_onset",
    ]
    _write_csv(cache_path, metric_rows, headers)
    return metric_rows


def _load_or_compute_metric_rows(bundle_dir: Path) -> List[Dict[str, object]]:
    loss_csv = bundle_dir / "evaluation" / "per_file_loss.csv"
    if not loss_csv.is_file():
        raise FileNotFoundError(f"Missing per_file_loss.csv: {loss_csv}")

    loss_rows = _read_csv(loss_csv)
    metrics_csv = bundle_dir / "evaluation" / "per_file_metrics.csv"

    if metrics_csv.is_file():
        cached_rows = _merge_rows(loss_rows, _read_csv(metrics_csv))
        if _metrics_cache_complete(cached_rows, expected_len=len(loss_rows)):
            return cached_rows

    return _compute_metric_rows(bundle_dir, loss_rows)


def _pack_label(summary: Dict[str, Any]) -> str:
    keys = summary.get("sample_pack_keys")
    if isinstance(keys, list) and len(keys) > 0:
        return ",".join(str(k) for k in keys)
    return "all"


def _summary_row(
    root: Path,
    bundle_dir: Path,
    summary: Dict[str, Any],
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    mss_mean, mss_std = _mean_std(_to_float(r.get("test/loss")) for r in rows)
    lsd_mean, lsd_std = _mean_std(_to_float(r.get("test/lsd")) for r in rows)
    sf_mean, sf_std = _mean_std(_to_float(r.get("test/flux_onset")) for r in rows)
    return {
        "bundle": _bundle_relpath(root, bundle_dir),
        "pack": _pack_label(summary),
        "items": len(rows),
        "mss_mean": mss_mean,
        "mss_std": mss_std,
        "lsd_mean": lsd_mean,
        "lsd_std": lsd_std,
        "sf_mean": sf_mean,
        "sf_std": sf_std,
    }


def _summary_markdown(summary_rows: List[Dict[str, object]]) -> str:
    rows = [
        [
            row["bundle"],
            row["pack"],
            row["items"],
            _fmt_float(_to_float(row["mss_mean"])),
            _fmt_float(_to_float(row["mss_std"])),
            _fmt_float(_to_float(row["lsd_mean"])),
            _fmt_float(_to_float(row["lsd_std"])),
            _fmt_float(_to_float(row["sf_mean"])),
            _fmt_float(_to_float(row["sf_std"])),
        ]
        for row in summary_rows
    ]
    return _markdown_table(SUMMARY_HEADERS, rows)


def _summary_latex(summary_rows: List[Dict[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{lllccc}",
        r"\hline",
        r"Bundle & Pack & N & MSS $\downarrow$ & LSD $\downarrow$ & SF $\downarrow$ \\",
        r"\hline",
    ]
    for row in summary_rows:
        bundle = _latex_escape(row["bundle"])
        pack = _latex_escape(row["pack"])
        n_items = _latex_escape(row["items"])
        mss = f"{_fmt_float(_to_float(row['mss_mean']), 3)} $\\pm$ {_fmt_float(_to_float(row['mss_std']), 4)}"
        lsd = f"{_fmt_float(_to_float(row['lsd_mean']), 3)} $\\pm$ {_fmt_float(_to_float(row['lsd_std']), 4)}"
        sf = f"{_fmt_float(_to_float(row['sf_mean']), 3)} $\\pm$ {_fmt_float(_to_float(row['sf_std']), 4)}"
        lines.append(f"{bundle} & {pack} & {n_items} & {mss} & {lsd} & {sf} \\\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _extreme_table_rows(selected: List[Tuple[Dict[str, object], float]]) -> List[List[object]]:
    rows: List[List[object]] = []
    for rank, (row, _) in enumerate(selected, start=1):
        rows.append(
            [
                rank,
                _sample_name(row),
                _fmt_float(_to_float(row.get("test/loss"))),
                _fmt_float(_to_float(row.get("test/lsd"))),
                _fmt_float(_to_float(row.get("test/flux_onset"))),
            ]
        )
    return rows


def _write_bundle_extremes(
    report_dir: Path,
    root: Path,
    bundle_dir: Path,
    summary: Dict[str, Any],
    rows: List[Dict[str, object]],
) -> str:
    relpath = _bundle_relpath(root, bundle_dir)
    safe_name = _safe_report_name(relpath)
    out_path = report_dir / f"extremes_{safe_name}.txt"

    lines = [
        f"bundle: {relpath}",
        f"pack: {_pack_label(summary)}",
        f"items: {len(rows)}",
        "",
    ]

    headers = ["rank", "sample", "MSS", "LSD", "SF"]
    for label, column in RANK_METRICS:
        values: List[Tuple[Dict[str, object], float]] = []
        for row in rows:
            value = _to_float(row.get(column))
            if _is_finite(value):
                values.append((row, value))

        lines.append(f"## {label} worst {TOP_K}")
        if len(values) == 0:
            lines.append("No finite values available.")
            lines.append("")
            continue

        descending = sorted(values, key=lambda item: item[1], reverse=True)[:TOP_K]
        lines.append(_markdown_table(headers, _extreme_table_rows(descending)))
        lines.append("")

        lines.append(f"## {label} best {TOP_K}")
        ascending = sorted(values, key=lambda item: item[1])[:TOP_K]
        lines.append(_markdown_table(headers, _extreme_table_rows(ascending)))
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path.name


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=str, help="Run directory or one exported bundle directory")
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory for generated reports. Defaults to <root>/reports or <bundle>/evaluation.",
    )
    parser.add_argument(
        "--tex-out",
        type=str,
        default=None,
        help="Optional explicit path for the summary TeX table.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    bundle_dirs = _bundle_dirs(root)
    bundle_dirs = sorted(bundle_dirs, key=lambda p: _bundle_relpath(root, p))

    if args.report_dir is not None:
        report_dir = Path(args.report_dir).resolve()
    elif len(bundle_dirs) == 1 and bundle_dirs[0] == root:
        report_dir = root / "evaluation"
    else:
        report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    extremes_index: List[Dict[str, object]] = []

    for bundle_dir in bundle_dirs:
        summary_path = bundle_dir / "evaluation" / "summary.json"
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        metric_rows = _load_or_compute_metric_rows(bundle_dir)
        summary_rows.append(_summary_row(root, bundle_dir, summary, metric_rows))
        report_name = _write_bundle_extremes(report_dir, root, bundle_dir, summary, metric_rows)
        extremes_index.append(
            {
                "bundle": _bundle_relpath(root, bundle_dir),
                "pack": _pack_label(summary),
                "extremes_report": report_name,
            }
        )

    summary_rows = sorted(summary_rows, key=lambda row: str(row["bundle"]))
    summary_table_txt = report_dir / "summary_table.txt"
    summary_table_txt.write_text(_summary_markdown(summary_rows) + "\n", encoding="utf-8")
    _write_csv(report_dir / "summary_table.csv", summary_rows, SUMMARY_HEADERS)
    tex_out = Path(args.tex_out).resolve() if args.tex_out is not None else report_dir / "summary_table.tex"
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.write_text(_summary_latex(summary_rows), encoding="utf-8")
    _write_csv(report_dir / "extremes_index.csv", extremes_index, ["bundle", "pack", "extremes_report"])

    print(f"[postprocess_results] bundles: {len(summary_rows)}")
    print(f"[postprocess_results] summary: {summary_table_txt}")
    print(f"[postprocess_results] tex: {tex_out}")
    print(f"[postprocess_results] reports: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
