#!/usr/bin/env python3
"""
Compile evaluation metrics into LaTeX tables.

This version reports per-run, per-sample statistics:
- MSS: mean/std across test samples (from per_file_loss.csv)
- LSD: mean/std across test samples (computed from recon/target wav pairs)
- SF:  mean/std across test samples (computed from recon/target wav pairs)

Important:
- std here is NOT across runs.
- std here is over test samples within a single run.
"""
import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio

MODELS = [
    "modal",
    "all_parallel",
    "noise_parallel_transient_params",
    "noise_transient_params",
    "noise_params",
    "transient_params",
]


def _to_float(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _is_finite(v: float) -> bool:
    return math.isfinite(v)


def _mean_std(values: List[float]) -> Tuple[float, float]:
    # ### HIGHLIGHT: std is computed over per-sample values within one run.
    xs = [x for x in values if _is_finite(x)]
    if len(xs) == 0:
        return float("nan"), float("nan")
    mean = float(statistics.fmean(xs))
    std = float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0
    return mean, std


def _parse_model_split(name: str) -> Tuple[Optional[str], str, str]:
    for model in sorted(MODELS, key=len, reverse=True):
        if not name.startswith(model):
            continue
        split = name[len(model) :].lstrip("_")
        instrument = "all"
        if split == "a":
            source = "acoustic"
        elif split == "e":
            source = "electronic"
        elif split in ("all", ""):
            source = "all"
        else:
            source = "all"
            instrument = split
        return model, source, instrument
    return None, "all", "all"


def _parse_model_from_config_stem(stem: str) -> Tuple[str, str, str]:
    clean = re.sub(r"^\d+_", "", stem)
    model, source, instrument = _parse_model_split(clean)
    if model is None:
        model = clean
    return model, source, instrument


def _pad_to_min_length(x: torch.Tensor, min_len: int) -> torch.Tensor:
    if x.shape[-1] >= min_len:
        return x
    return torch.nn.functional.pad(x, (0, int(min_len - x.shape[-1])))


def _lsd_single(x: torch.Tensor, y: torch.Tensor, n_fft: int = 8092, hop: int = 64) -> float:
    # x,y: [1,T]
    eps = 1e-8
    min_len = max(int(n_fft), int(n_fft // 2 + 1))
    x = _pad_to_min_length(x, min_len)
    y = _pad_to_min_length(y, min_len)
    win = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True, pad_mode="constant")
    Y = torch.stft(y, n_fft=n_fft, hop_length=hop, window=win, return_complex=True, pad_mode="constant")
    X = torch.log(torch.square(torch.abs(X)) + eps)
    Y = torch.log(torch.square(torch.abs(Y)) + eps)
    lsd = torch.mean(torch.square(X - Y), dim=-2)
    lsd = torch.mean(torch.sqrt(lsd), dim=-1)
    return float(lsd.squeeze(0).cpu())


def _sf_single(x: torch.Tensor, y: torch.Tensor, n_fft: int = 1024, hop: int = 64) -> float:
    # x,y: [1,T]
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

    # Keep original implementation behavior (diff over dim=1).
    flux_x = torch.diff(torch.abs(X), dim=1)
    flux_y = torch.diff(torch.abs(Y), dim=1)
    flux_x = (flux_x + torch.abs(flux_x)) / 2
    flux_y = (flux_y + torch.abs(flux_y)) / 2
    flux_x = torch.square(flux_x).sum(dim=1)
    flux_y = torch.square(flux_y).sum(dim=1)
    onset_err = torch.mean(torch.abs(flux_x - flux_y), dim=-1)
    return float(onset_err.squeeze(0).cpu())


def _compute_or_load_per_file_metrics(run_dir: Path) -> List[Dict[str, float]]:
    eval_dir = run_dir / "evaluation"
    per_file_loss = eval_dir / "per_file_loss.csv"
    if not per_file_loss.is_file():
        raise FileNotFoundError(f"Missing per_file_loss.csv: {per_file_loss}")

    cache_csv = eval_dir / "per_file_metrics.csv"
    if cache_csv.is_file():
        rows = []
        with cache_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(
                    {
                        "test/loss": _to_float(r.get("test/loss")),
                        "test/lsd": _to_float(r.get("test/lsd")),
                        "test/flux_onset": _to_float(r.get("test/flux_onset")),
                    }
                )
        if len(rows) > 0:
            return rows

    # ### HIGHLIGHT: Build per-file metrics cache from recon/target audio.
    out_rows: List[Dict[str, float]] = []
    recon_root = run_dir / "recon"
    target_root = run_dir / "target"
    if not recon_root.exists() or not target_root.exists():
        raise FileNotFoundError(
            f"Missing recon/target directories under {run_dir}. "
            "Cannot compute per-sample LSD/SF."
        )

    with per_file_loss.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            src_rel = str(r["source_filename"]).replace("\\", "/")
            recon_path = recon_root / src_rel
            target_path = target_root / src_rel
            if not recon_path.is_file() or not target_path.is_file():
                continue

            recon, _ = torchaudio.load(str(recon_path))
            target, _ = torchaudio.load(str(target_path))
            recon = recon[:1, :]
            target = target[:1, :]

            x = recon.squeeze(0).unsqueeze(0)
            y = target.squeeze(0).unsqueeze(0)

            lsd = _lsd_single(x, y)
            sf = _sf_single(x, y)
            loss_v = _to_float(r.get("test/loss"))

            out_rows.append(
                {
                    "test/loss": loss_v,
                    "test/lsd": lsd,
                    "test/flux_onset": sf,
                }
            )

    with cache_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test/loss", "test/lsd", "test/flux_onset"])
        for rr in out_rows:
            w.writerow([rr["test/loss"], rr["test/lsd"], rr["test/flux_onset"]])

    return out_rows


def _model_label(model: str) -> str:
    mapping = {
        "all_parallel": "All Parallel",
        "noise_parallel_transient_params": "Noise+Transient Parallel",
        "noise_transient_params": "Noise+Transient",
        "noise_params": "Noise Params",
        "transient_params": "Transient Params",
        "modal": "Modal",
    }
    return mapping.get(model, model.replace("_", " "))


def _fmt_mean(x: float) -> str:
    if not _is_finite(x):
        return "-"
    return f"{x:.3f}"


def _fmt_std(x: float) -> str:
    if not _is_finite(x):
        return "-"
    return f"{x:.4f}"


def _cell_mean_std(mean: float, std: float, is_best: bool) -> str:
    if not _is_finite(mean):
        return "-"
    mean_s = _fmt_mean(mean)
    std_s = _fmt_std(std if _is_finite(std) else 0.0)
    if is_best:
        mean_s = r"\textbf{" + mean_s + "}"
    return mean_s + r" {\scriptsize$\pm$ " + std_s + "}"


def _cell_loss_std(std: float, is_best: bool) -> str:
    if not _is_finite(std):
        return "-"
    s = _fmt_std(std)
    if is_best:
        s = r"\textbf{" + s + "}"
    return s


def _load_run_rows(indir: Path, run_name_regex: Optional[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    regex = re.compile(run_name_regex) if run_name_regex else None

    for f in indir.rglob("summary.json"):
        if not f.is_file() or f.parent.name != "evaluation":
            continue
        run_dir = f.parents[1]
        run_name = run_dir.name
        if regex and regex.search(run_name) is None:
            continue

        payload = json.loads(f.read_text(encoding="utf-8"))
        config = str(payload.get("config", ""))
        config_stem = Path(config).stem if config else run_name
        model, source, instrument = _parse_model_from_config_stem(config_stem)

        rows.append(
            {
                "run_name": run_name,
                "run_dir": run_dir,
                "model": model,
                "source": source,
                "instrument": instrument,
            }
        )

    if len(rows) == 0:
        raise RuntimeError(f"No usable runs found under: {indir}")
    return rows


def _pick_single_run_per_model(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        model = str(r["model"])
        grouped.setdefault(model, []).append(r)

    out: List[Dict[str, object]] = []
    for model in MODELS + sorted(set(grouped.keys()) - set(MODELS)):
        if model not in grouped:
            continue
        candidates = sorted(grouped[model], key=lambda x: str(x["run_name"]))
        # If multiple runs remain, pick lexicographically first for determinism.
        out.append(candidates[0])
    return out


def _write_latex_table(outfile, header: List[str], body: List[List[str]]) -> None:
    if len(header) == 0:
        outfile.write("% empty table\n")
        return
    align = "l" + "c" * (len(header) - 1)
    outfile.write(r"\begin{tabular}{" + align + "}" + "\n")
    outfile.write(r"\hline" + "\n")
    outfile.write(" & ".join(header) + r" \\" + "\n")
    outfile.write(r"\hline" + "\n")
    for row in body:
        outfile.write(" & ".join(row) + r" \\" + "\n")
    outfile.write(r"\hline" + "\n")
    outfile.write(r"\end{tabular}" + "\n")


def _build_all_table(selected_runs: List[Dict[str, object]]) -> Tuple[List[str], List[List[str]]]:
    header = ["Method", r"MSS $\downarrow$", r"LSD $\downarrow$", r"SF $\downarrow$", r"LossStd $\downarrow$"]

    stats_per_model: Dict[str, Dict[str, float]] = {}
    for rr in selected_runs:
        model = str(rr["model"])
        metrics_rows = _compute_or_load_per_file_metrics(Path(rr["run_dir"]))

        mss_vals = [float(x["test/loss"]) for x in metrics_rows if _is_finite(float(x["test/loss"]))]
        lsd_vals = [float(x["test/lsd"]) for x in metrics_rows if _is_finite(float(x["test/lsd"]))]
        sf_vals = [
            float(x["test/flux_onset"]) for x in metrics_rows if _is_finite(float(x["test/flux_onset"]))
        ]

        mss_mean, mss_std = _mean_std(mss_vals)
        lsd_mean, lsd_std = _mean_std(lsd_vals)
        sf_mean, sf_std = _mean_std(sf_vals)

        stats_per_model[model] = {
            "mss_mean": mss_mean,
            "mss_std": mss_std,
            "lsd_mean": lsd_mean,
            "lsd_std": lsd_std,
            "sf_mean": sf_mean,
            "sf_std": sf_std,
            "loss_std": mss_std,
        }

    def _best(metric_mean_key: str) -> float:
        vals = [v[metric_mean_key] for v in stats_per_model.values() if _is_finite(v[metric_mean_key])]
        return min(vals) if len(vals) > 0 else float("nan")

    best_mss = _best("mss_mean")
    best_lsd = _best("lsd_mean")
    best_sf = _best("sf_mean")
    best_loss_std = _best("loss_std")

    body: List[List[str]] = []
    ordered_models = [m for m in MODELS if m in stats_per_model] + sorted(
        set(stats_per_model.keys()) - set(MODELS)
    )
    for model in ordered_models:
        s = stats_per_model[model]
        body.append(
            [
                _model_label(model),
                _cell_mean_std(s["mss_mean"], s["mss_std"], abs(s["mss_mean"] - best_mss) < 1e-12),
                _cell_mean_std(s["lsd_mean"], s["lsd_std"], abs(s["lsd_mean"] - best_lsd) < 1e-12),
                _cell_mean_std(s["sf_mean"], s["sf_std"], abs(s["sf_mean"] - best_sf) < 1e-12),
                _cell_loss_std(s["loss_std"], abs(s["loss_std"] - best_loss_std) < 1e-12),
            ]
        )

    return header, body


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("indir", type=str, help="Input directory")
    parser.add_argument("type", type=str, choices=["all", "instrument"])
    parser.add_argument(
        "--input-format",
        choices=["auto", "summary_json", "metrics_csv"],
        default="auto",
    )
    parser.add_argument(
        "--run-name-regex",
        type=str,
        default=None,
        help="Only include runs whose directory name matches this regex.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default=sys.stdout,
        type=argparse.FileType("w", encoding="utf-8"),
    )
    args = parser.parse_args(argv)

    indir = Path(args.indir)
    run_rows = _load_run_rows(indir, run_name_regex=args.run_name_regex)

    # Keep only "all/all" rows for this table mode and choose one run per model.
    if args.type == "all":
        run_rows = [r for r in run_rows if r["instrument"] == "all" and r["source"] == "all"]
        run_rows = _pick_single_run_per_model(run_rows)
        header, body = _build_all_table(run_rows)
        print(f"[compile_results] selected runs: {len(run_rows)}")
        _write_latex_table(args.outfile, header, body)
        return 0

    # Instrument mode remains unsupported for unlabeled custom dataset workflow.
    print("[compile_results] selected runs: 0")
    args.outfile.write("% empty table\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
