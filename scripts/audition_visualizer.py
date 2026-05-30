#!/usr/bin/env python3
"""
Interactive HTML audition/visualization server for reconstruction bundles.

Features:
- Randomly shows N samples (default 10) from a result folder
- Plays target/reconstruction audio
- Displays high-resolution waveform and spectrogram for target/reconstruction
- Shows per-sample metrics (from per_file_loss.csv + optional per_file_metrics.csv)
- Refresh button to load a new random set

Expected folder layout (from export_recon_wavs.py):
  <result_dir>/
    manifest.csv
    recon/<source_filename>.wav
    target/<source_filename>.wav                # optional
    evaluation/per_file_loss.csv
    evaluation/per_file_metrics.csv             # optional
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import inspect
import io
import json
import random
import socketserver
import sys
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib

# ### HIGHLIGHT: Use non-interactive backend for server-side image rendering.
matplotlib.use("Agg")
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer

from drumblender.utils.modal_analysis import CQTModalAnalysis as ModalAnalysis

try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None

try:
    from scipy.io import wavfile as scipy_wavfile  # type: ignore
except Exception:
    scipy_wavfile = None


DEFAULT_RESULT_DIR = "../results/run_NOISEDAC_20260414_111726"
DEFAULT_COMPARE_RESULT_DIR = "../results/run_NOISEDAC_20260412_231956"
DEFAULT_DATASET_DIR_0412 = "../datasets"
DEFAULT_DATASET_DIR_0414 = "../datasets_NEW"
PLOT_CACHE_VERSION = "v7-audio-comparison"
MODAL_SAMPLE_RATE = 48000
MODAL_FMIN = 20.0
MODAL_N_BINS = 240
MODAL_BINS_PER_OCTAVE = 24
MODAL_HOP = 256
MODAL_MIN_LENGTH = 10
MODAL_THRESHOLD_DB = -80.0
MODAL_DIFF_THRESHOLD = 5.0
MODAL_P_KERNEL = 31
MODAL_MAX_GAP = 2
MODAL_MIN_ACTIVE_RATIO = 0.25
MODAL_MIN_TRACK_ENERGY = 0.0
MODAL_FMAX = 22050.0
MODAL_LOG_FREQ_MIN = 20.0
MODAL_DYNAMIC_RANGE_DB = 100.0
MODAL_MIN_SAMPLES = 131072


@dataclass
class SampleRow:
    index: int
    meta_key: str
    source_filename: str
    length: int
    recon_path: Path
    target_path: Optional[Path]
    metrics: Dict[str, float]


@dataclass
class RunView:
    label: str
    result_dir: Path
    rows: List[SampleRow]
    by_meta_key: Dict[str, SampleRow]
    by_source_filename: Dict[str, SampleRow]
    dataset_dir: Optional[Path]
    feature_by_key: Dict[str, Dict]
    feature_by_relpath: Dict[str, Dict]


def _to_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _finite(x: float) -> bool:
    return bool(np.isfinite(x))


def _safe_rel(path_str: str) -> str:
    return str(Path(path_str.replace("\\", "/")))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _normalize_relpath(value: str) -> str:
    return str(value).replace("\\", "/").strip().lower()


def _resolve_result_dir(path: Path) -> Path:
    """
    Accept either an exported bundle directory or a top-level run result directory.
    """
    path = path.expanduser().resolve()
    if (path / "manifest.csv").is_file():
        return path

    candidates: List[Path] = []
    candidates.extend(sorted((path / "all").glob("*")))
    candidates.extend(sorted((path / "per_pack").glob("*")))
    candidates.extend(p.parent for p in sorted(path.glob("**/manifest.csv")))

    seen: set[str] = set()
    unique: List[Path] = []
    for candidate in candidates:
        if not (candidate / "manifest.csv").is_file():
            continue
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    if not unique:
        raise FileNotFoundError(f"No manifest.csv found under: {path}")

    for candidate in unique:
        if candidate.parent.name == "all" or candidate.name.endswith("_all"):
            return candidate.resolve()
    return unique[0].resolve()


def _resolve_dataset_dir(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    root = Path(path).expanduser().resolve()
    candidates = [
        root,
        root / "modal_features" / "processed_modal_flat",
        root / "processed_modal_flat",
    ]
    for candidate in candidates:
        if (candidate / "metadata.json").is_file() and (candidate / "features").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find modal feature dataset under {root}. "
        "Expected metadata.json and features/."
    )


def _load_feature_index(dataset_dir: Optional[Path]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    if dataset_dir is None:
        return {}, {}
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    by_key: Dict[str, Dict] = {}
    by_relpath: Dict[str, Dict] = {}
    for key, item in metadata.items():
        if not isinstance(item, dict):
            continue
        item = dict(item)
        item["_key"] = str(key)
        by_key[str(key)] = item
        rel = _normalize_relpath(str(item.get("orig_relpath", "")))
        if rel:
            by_relpath[rel] = item
    return by_key, by_relpath


def _dataset_feature_path(dataset_dir: Path, item: Dict) -> Optional[Path]:
    feature_file = item.get("feature_file")
    if not feature_file:
        return None
    path = dataset_dir / str(feature_file)
    return path if path.is_file() else None


def _auto_dataset_dir_for_result(result_dir: Path, fallback: Optional[str] = None) -> Optional[Path]:
    text = str(result_dir).replace("\\", "/")
    if "20260414" in text:
        return _resolve_dataset_dir(DEFAULT_DATASET_DIR_0414)
    if "20260412" in text:
        return _resolve_dataset_dir(DEFAULT_DATASET_DIR_0412)
    return _resolve_dataset_dir(fallback)


def _label_from_result_dir(result_dir: Path) -> str:
    text = str(result_dir).replace("\\", "/")
    if "20260414" in text:
        return "NOISEDAC"
    if "20260412" in text:
        return "Ref. NOISEDAC"
    return result_dir.name


def _row_maps(rows: Sequence[SampleRow]) -> Tuple[Dict[str, SampleRow], Dict[str, SampleRow]]:
    by_meta_key = {str(row.meta_key): row for row in rows if row.meta_key}
    by_source = {_normalize_relpath(row.source_filename): row for row in rows}
    return by_meta_key, by_source


def _match_row(base_row: SampleRow, run: RunView) -> Optional[SampleRow]:
    row = run.by_meta_key.get(str(base_row.meta_key))
    if row is not None:
        return row
    return run.by_source_filename.get(_normalize_relpath(base_row.source_filename))


def _build_run_view(result_input: str, dataset_input: Optional[str] = None) -> RunView:
    result_dir = _resolve_result_dir(Path(result_input))
    dataset_dir = _resolve_dataset_dir(dataset_input) if dataset_input else None
    rows = _collect_rows(result_dir)
    by_meta_key, by_source = _row_maps(rows)
    feature_by_key, feature_by_relpath = _load_feature_index(dataset_dir)
    return RunView(
        label=_label_from_result_dir(result_dir),
        result_dir=result_dir,
        rows=rows,
        by_meta_key=by_meta_key,
        by_source_filename=by_source,
        dataset_dir=dataset_dir,
        feature_by_key=feature_by_key,
        feature_by_relpath=feature_by_relpath,
    )


def _merge_metrics(
    loss_rows: Optional[List[Dict[str, str]]],
    metric_rows: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """
    Merge per-file loss/metric tables.
    Priority:
    1) key-based merge using common identity column
    2) row-order merge fallback
    """
    if not loss_rows and not metric_rows:
        return []

    if loss_rows is None:
        return metric_rows or []
    if metric_rows is None:
        return loss_rows

    key_candidates = ("index", "meta_key", "source_filename")
    chosen = None
    if len(loss_rows) > 0 and len(metric_rows) > 0:
        for k in key_candidates:
            if k in loss_rows[0] and k in metric_rows[0]:
                chosen = k
                break

    if chosen is not None:
        by_key: Dict[str, Dict[str, str]] = {}
        for row in loss_rows:
            kv = str(row.get(chosen, "")).strip()
            if kv:
                by_key[kv] = row
        out: List[Dict[str, str]] = []
        for row in metric_rows:
            kv = str(row.get(chosen, "")).strip()
            merged = {}
            if kv and kv in by_key:
                merged.update(by_key[kv])
            merged.update(row)
            out.append(merged)
        return out

    n = min(len(loss_rows), len(metric_rows))
    out = []
    for i in range(n):
        merged = {}
        merged.update(loss_rows[i])
        merged.update(metric_rows[i])
        out.append(merged)
    return out


def _collect_rows(result_dir: Path) -> List[SampleRow]:
    manifest = result_dir / "manifest.csv"
    eval_dir = result_dir / "evaluation"
    loss_csv = eval_dir / "per_file_loss.csv"
    metrics_csv = eval_dir / "per_file_metrics.csv"

    if not manifest.is_file():
        raise FileNotFoundError(f"manifest.csv not found: {manifest}")

    manifest_rows = _read_csv(manifest)
    loss_rows = _read_csv(loss_csv) if loss_csv.is_file() else None
    metric_rows = _read_csv(metrics_csv) if metrics_csv.is_file() else None
    merged_metric_rows = _merge_metrics(loss_rows, metric_rows)

    # ### HIGHLIGHT: Build fast lookup for metric rows.
    metric_by_key: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for r in merged_metric_rows:
        k = (
            str(r.get("index", "")),
            str(r.get("meta_key", "")),
            _safe_rel(str(r.get("source_filename", ""))),
        )
        metric_by_key[k] = r

    rows: List[SampleRow] = []
    for m in manifest_rows:
        idx_str = str(m.get("index", "")).strip()
        meta_key = str(m.get("meta_key", "")).strip()
        src = _safe_rel(str(m.get("source_filename", "")).strip())
        length_i = int(_to_float(m.get("length", 0)))

        recon_path_raw = str(m.get("recon_path", "")).strip()
        target_path_raw = str(m.get("target_path", "")).strip()

        recon_path = Path(recon_path_raw)
        if not recon_path.is_absolute():
            recon_path = result_dir / recon_path
        if not recon_path.exists():
            # fallback: map by source filename under recon/
            recon_path = result_dir / "recon" / Path(src)

        target_path: Optional[Path] = None
        if target_path_raw:
            tp = Path(target_path_raw)
            if not tp.is_absolute():
                tp = result_dir / tp
            if tp.exists():
                target_path = tp
        if target_path is None:
            tp2 = result_dir / "target" / Path(src)
            if tp2.exists():
                target_path = tp2

        k = (idx_str, meta_key, src)
        metric_row = metric_by_key.get(k, {})
        metric_dict: Dict[str, float] = {}
        for kk, vv in metric_row.items():
            if kk in ("index", "meta_key", "source_filename", "length"):
                continue
            fx = _to_float(vv)
            if _finite(fx):
                metric_dict[kk] = fx

        rows.append(
            SampleRow(
                index=int(_to_float(idx_str)),
                meta_key=meta_key,
                source_filename=src,
                length=length_i,
                recon_path=recon_path,
                target_path=target_path,
                metrics=metric_dict,
            )
        )

    # Keep only rows with valid recon audio files.
    rows = [r for r in rows if r.recon_path.is_file()]
    if len(rows) == 0:
        raise RuntimeError("No valid recon audio rows found.")
    return rows


def _slugify_metric(name: str) -> str:
    name = name.replace("test/", "")
    name = name.replace("flux_onset", "SF")
    name = name.replace("lsd", "LSD")
    name = name.replace("loss", "MSS/Loss")
    name = name.replace("mss_sc", "MSS_SC")
    name = name.replace("mss_log", "MSS_LOG")
    return name


def _load_mono(path: Path) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono float32 using robust backend fallback.
    Priority:
      1) torchaudio
      2) soundfile
      3) scipy.io.wavfile
    This avoids Windows torchcodec/ffmpeg dependency issues.
    """
    # 1) torchaudio path (fast in torch-enabled environments)
    try:
        wav, sr = torchaudio.load(str(path))
        wav_np = wav[:1, :].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        return wav_np, int(sr)
    except Exception:
        pass

    # 2) soundfile fallback (handles float WAV well on Windows)
    if sf is not None:
        try:
            data, sr = sf.read(str(path), always_2d=True)
            mono = data[:, :1].squeeze(1).astype(np.float32, copy=False)
            return mono, int(sr)
        except Exception:
            pass

    # 3) scipy fallback
    if scipy_wavfile is not None:
        try:
            sr, data = scipy_wavfile.read(str(path))
            if data.ndim == 2:
                data = data[:, 0]
            if np.issubdtype(data.dtype, np.integer):
                maxv = np.iinfo(data.dtype).max
                data = data.astype(np.float32) / max(maxv, 1)
            else:
                data = data.astype(np.float32, copy=False)
            return data, int(sr)
        except Exception:
            pass

    raise RuntimeError(
        f"Failed to read audio: {path}. Install one of: torchaudio-compatible torchcodec,"
        " soundfile, or scipy."
    )


def _plot_waveform_png(audio: np.ndarray, sr: int, title: str) -> bytes:
    # ### HIGHLIGHT: High-resolution waveform render for presentation quality.
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=180)
    t = np.arange(audio.shape[0], dtype=np.float64) / max(sr, 1)
    ax.plot(t, audio, linewidth=0.9, color="#0f766e")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amp", fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlim(0, t[-1] if t.size > 0 else 1.0)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _frequency_edges_hz(freqs_hz: torch.Tensor) -> torch.Tensor:
    if freqs_hz.numel() < 2:
        return torch.tensor([1e-6, float(freqs_hz.item()) + 1.0])

    edges = torch.empty(freqs_hz.numel() + 1, dtype=freqs_hz.dtype)
    if float(freqs_hz[0]) > 0:
        edges[1:-1] = torch.sqrt(freqs_hz[:-1] * freqs_hz[1:])
        first_ratio = freqs_hz[1] / freqs_hz[0]
        last_ratio = freqs_hz[-1] / freqs_hz[-2]
        edges[0] = freqs_hz[0] / torch.sqrt(first_ratio)
        edges[-1] = freqs_hz[-1] * torch.sqrt(last_ratio)
    else:
        edges[1:-1] = (freqs_hz[:-1] + freqs_hz[1:]) * 0.5
        edges[0] = max(float(freqs_hz[1]) * 0.5, 1e-6)
        edges[-1] = freqs_hz[-1] + (freqs_hz[-1] - freqs_hz[-2]) * 0.5
    return edges


def _modal_kwargs() -> Dict[str, object]:
    values = {
        "sample_rate": MODAL_SAMPLE_RATE,
        "hop_length": MODAL_HOP,
        "fmin": MODAL_FMIN,
        "n_bins": MODAL_N_BINS,
        "bins_per_octave": MODAL_BINS_PER_OCTAVE,
        "min_length": MODAL_MIN_LENGTH,
        "threshold": MODAL_THRESHOLD_DB,
        "diff_threshold": MODAL_DIFF_THRESHOLD,
        "p_kernel": MODAL_P_KERNEL,
        "max_gap": MODAL_MAX_GAP,
        "min_active_ratio": MODAL_MIN_ACTIVE_RATIO,
        "min_track_energy": MODAL_MIN_TRACK_ENERGY,
    }
    params = inspect.signature(ModalAnalysis.__init__).parameters
    return {key: value for key, value in values.items() if key in params}


def _modal_analysis_spectrogram_db(
    audio: np.ndarray,
    sr: int,
    duration: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    # This mirrors plot_modal_reconstruction_compare.py's original/recon
    # spectrogram path, using the same CQTModalAnalysis implementation as
    # scripts/build_modal_features.py.
    waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    original_duration = audio.shape[0] / max(sr, 1)
    if sr != MODAL_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(int(sr), MODAL_SAMPLE_RATE)(waveform)
        sr = MODAL_SAMPLE_RATE
    if duration is None:
        duration = original_duration
    if waveform.shape[-1] < MODAL_MIN_SAMPLES:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, MODAL_MIN_SAMPLES - waveform.shape[-1]),
        )

    modal = ModalAnalysis(**_modal_kwargs())
    with torch.no_grad():
        mag = modal.spectrogram(waveform, complex=False).squeeze(0).clamp_min(1e-8)
    db = 20.0 * torch.log10(mag)
    freqs_hz = torch.as_tensor(modal.frequencies(), dtype=db.dtype, device=db.device)

    max_frame = min(db.shape[1], max(1, int(duration * MODAL_SAMPLE_RATE / MODAL_HOP) + 1))
    keep_freq = freqs_hz <= min(MODAL_FMAX, MODAL_SAMPLE_RATE / 2.0)
    db = db[keep_freq, :max_frame].cpu()
    freqs_hz = freqs_hz[keep_freq].cpu()
    return db, freqs_hz, float(duration)


def _plot_modal_spectrogram_png(
    db: torch.Tensor,
    freqs_hz: torch.Tensor,
    duration: float,
    title: str,
    vmin: float,
    vmax: float,
) -> bytes:
    lower_hz = max(MODAL_LOG_FREQ_MIN, 1e-6)
    upper_hz = min(MODAL_FMAX, MODAL_SAMPLE_RATE / 2.0)
    freq_edges = _frequency_edges_hz(freqs_hz)
    time_edges = torch.linspace(0.0, duration, db.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=180)
    im = ax.pcolormesh(
        time_edges.numpy(),
        freq_edges.numpy(),
        db.numpy(),
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_yscale("log")
    ax.set_xlim(0.0, duration)
    ax.set_ylim(lower_hz, upper_hz)
    yticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    yticks = [tick for tick in yticks if lower_hz <= tick <= upper_hz]
    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(tick) for tick in yticks])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.set_ylabel("dB", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _modal_feature_freq_amp(feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    feat = feature.detach().float().cpu()
    if feat.ndim == 4 and feat.shape[1] == 1:
        feat = feat.squeeze(1)
    if feat.ndim != 3 or feat.shape[0] < 2:
        raise RuntimeError(f"Expected modal feature shape [3, modes, frames], got {tuple(feat.shape)}")
    freqs = feat[0]
    amps = feat[1].clamp_min(0.0)
    if freqs.numel() > 0 and float(freqs.max().item()) <= float(2.0 * np.pi + 1e-3):
        freqs = freqs * (MODAL_SAMPLE_RATE / float(2.0 * np.pi))
    return freqs, amps


def _modal_feature_db_values(feature: torch.Tensor) -> Optional[torch.Tensor]:
    _, amps = _modal_feature_freq_amp(feature)
    mask = amps > 0
    if not bool(mask.any()):
        return None
    return 20.0 * torch.log10(amps[mask].clamp_min(1e-8))


def _plot_modal_feature_png(
    feature: torch.Tensor,
    duration: float,
    title: str,
    vmin: float,
    vmax: float,
) -> bytes:
    freqs_hz, amps = _modal_feature_freq_amp(feature)
    lower_hz = max(MODAL_LOG_FREQ_MIN, 1e-6)
    upper_hz = min(MODAL_FMAX, MODAL_SAMPLE_RATE / 2.0)
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=180)
    ax.set_facecolor("black")

    collection = None
    if freqs_hz.numel() > 0 and amps.numel() > 0 and freqs_hz.shape[1] >= 2:
        frame_times = torch.linspace(0.0, float(duration), freqs_hz.shape[1])
        t0 = frame_times[:-1].unsqueeze(0).expand(freqs_hz.shape[0], -1)
        t1 = frame_times[1:].unsqueeze(0).expand(freqs_hz.shape[0], -1)
        f0 = freqs_hz[:, :-1]
        f1 = freqs_hz[:, 1:]
        a0 = amps[:, :-1]
        a1 = amps[:, 1:]
        mask = (
            ((a0 > 0) | (a1 > 0))
            & (f0 > 0)
            & (f1 > 0)
            & (f0 >= lower_hz)
            & (f0 <= upper_hz)
            & (f1 >= lower_hz)
            & (f1 <= upper_hz)
            & (t0 <= duration)
        )
        if bool(mask.any()):
            segments = torch.stack(
                [
                    torch.stack([t0[mask], f0[mask]], dim=1),
                    torch.stack([t1[mask].clamp_max(duration), f1[mask]], dim=1),
                ],
                dim=1,
            )
            segment_amp = ((a0[mask] + a1[mask]) * 0.5).clamp_min(1e-8)
            segment_db = 20.0 * torch.log10(segment_amp)
            collection = LineCollection(
                segments.numpy(),
                cmap="magma",
                norm=Normalize(vmin=vmin, vmax=vmax),
                linewidths=1.0,
                alpha=0.95,
                capstyle="round",
                joinstyle="round",
            )
            collection.set_array(segment_db.numpy())
            ax.add_collection(collection)

    if collection is None:
        ax.text(0.5, 0.5, "No active modal feature tracks", color="white", ha="center", va="center", transform=ax.transAxes)
        mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="magma")
    else:
        mappable = collection

    ax.set_yscale("log")
    ax.set_xlim(0.0, duration)
    ax.set_ylim(lower_hz, upper_hz)
    yticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    yticks = [tick for tick in yticks if lower_hz <= tick <= upper_hz]
    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(tick) for tick in yticks])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.01)
    cbar.ax.set_ylabel("dB", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _plot_spectrogram_png(audio: np.ndarray, sr: int, title: str) -> bytes:
    db, freqs_hz, duration = _modal_analysis_spectrogram_db(audio, sr)
    vmax = float(db.max().item())
    vmin = vmax - MODAL_DYNAMIC_RANGE_DB
    return _plot_modal_spectrogram_png(db, freqs_hz, duration, title, vmin, vmax)


def _b64_png(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _metric_table_html(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "<div class='muted'>No per-file metrics found.</div>"
    parts = ["<table class='metric-table'><tbody>"]
    for k in sorted(metrics.keys()):
        parts.append(
            f"<tr><td>{html.escape(_slugify_metric(k))}</td><td>{metrics[k]:.6f}</td></tr>"
        )
    parts.append("</tbody></table>")
    return "".join(parts)


def _metric_compare_table_html(model_rows: Sequence[Tuple[str, Optional[SampleRow]]]) -> str:
    metric_names = sorted({name for _, row in model_rows if row is not None for name in row.metrics.keys()})
    if not metric_names:
        return "<div class='muted'>No per-file metrics found.</div>"

    parts = ["<table class='metric-table metric-compare'><thead><tr><th>Metric</th>"]
    for label, _ in model_rows:
        parts.append(f"<th>{html.escape(label)}</th>")
    parts.append("</tr></thead><tbody>")

    for name in metric_names:
        values: List[Optional[float]] = []
        for _, row in model_rows:
            values.append(None if row is None else row.metrics.get(name))
        parts.append(f"<tr><td>{html.escape(_slugify_metric(name))}</td>")
        for value in values:
            cell = "n/a" if value is None or not _finite(float(value)) else f"{float(value):.6f}"
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _row_search_text(row: SampleRow) -> str:
    metric_names = " ".join(row.metrics.keys())
    return " ".join(
        [
            str(row.index),
            row.meta_key,
            Path(row.source_filename).name,
            row.recon_path.name,
            row.target_path.name if row.target_path is not None else "",
            metric_names,
        ]
    ).lower()


def _filter_rows(rows: Sequence[SampleRow], query: str) -> List[SampleRow]:
    terms = [term for term in query.lower().split() if term]
    if not terms:
        return list(rows)
    out: List[SampleRow] = []
    for row in rows:
        haystack = _row_search_text(row)
        if all(term in haystack for term in terms):
            out.append(row)
    return out


def _rows_search_json(rows: Sequence[SampleRow]) -> str:
    payload = [
        {
            "index": row.index,
            "meta_key": row.meta_key,
            "source_display": Path(row.source_filename).name or row.source_filename,
            "haystack": _row_search_text(row),
        }
        for row in rows
    ]
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


class AuditionHandler(BaseHTTPRequestHandler):
    rows: List[SampleRow] = []
    runs: List[RunView] = []
    n_show: int = 10
    cache_dir: Path

    def _feature_item_for_row(self, row: SampleRow, run: RunView) -> Optional[Dict]:
        if run.dataset_dir is None:
            return None
        item = run.feature_by_key.get(str(row.meta_key))
        if item is not None:
            return item
        return run.feature_by_relpath.get(_normalize_relpath(row.source_filename))

    def _render_png_cached(self, audio_path: Path, kind: str) -> bytes:
        stat = audio_path.stat()
        key = f"{PLOT_CACHE_VERSION}::{audio_path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}::{kind}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        out_path = self.cache_dir / f"{digest}.png"
        if out_path.is_file():
            return out_path.read_bytes()

        audio, sr = _load_mono(audio_path)
        if kind == "wave":
            png = _plot_waveform_png(audio, sr, f"Waveform | {audio_path.name}")
        elif kind == "spec":
            png = _plot_spectrogram_png(audio, sr, f"Spectrogram | {audio_path.name}")
        else:
            raise ValueError(f"Unknown kind: {kind}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png)
        return png

    def _render_spec_pair_cached(self, row: SampleRow, run: RunView) -> Tuple[Optional[bytes], bytes]:
        target_path = row.target_path if row.target_path is not None and row.target_path.is_file() else None
        feature_item = self._feature_item_for_row(row, run)
        feature_path = (
            _dataset_feature_path(run.dataset_dir, feature_item)
            if run.dataset_dir is not None and feature_item is not None
            else None
        )
        recon_stat = row.recon_path.stat()
        parts = [
            PLOT_CACHE_VERSION,
            "spec-pair",
            run.label,
            str(row.recon_path.resolve()),
            str(recon_stat.st_mtime_ns),
            str(recon_stat.st_size),
        ]
        if target_path is not None:
            target_stat = target_path.stat()
            parts.extend(
                [
                    str(target_path.resolve()),
                    str(target_stat.st_mtime_ns),
                    str(target_stat.st_size),
                ]
            )
        else:
            parts.append("no-target")
        if feature_path is not None:
            feature_stat = feature_path.stat()
            parts.extend(
                [
                    str(run.dataset_dir.resolve() if run.dataset_dir is not None else ""),
                    str(feature_path.resolve()),
                    str(feature_stat.st_mtime_ns),
                    str(feature_stat.st_size),
                ]
            )
        else:
            parts.append("no-feature")

        digest = hashlib.md5("::".join(parts).encode("utf-8")).hexdigest()
        target_out = self.cache_dir / f"{digest}-original-spec.png"
        recon_out = self.cache_dir / f"{digest}-reconstruction-spec.png"
        has_original_plot = feature_path is not None or target_path is not None
        if recon_out.is_file() and (not has_original_plot or target_out.is_file()):
            target_png = target_out.read_bytes() if has_original_plot else None
            return target_png, recon_out.read_bytes()

        recon_audio, recon_sr = _load_mono(row.recon_path)
        target_png: Optional[bytes] = None
        if feature_path is not None:
            duration = float(feature_item.get("num_samples", row.length) or row.length) / MODAL_SAMPLE_RATE
            feature = torch.load(feature_path, map_location="cpu")
            recon_db, recon_freqs, _ = _modal_analysis_spectrogram_db(
                recon_audio,
                recon_sr,
                duration=duration,
            )
            vmax_candidates = [float(recon_db.max().item())]
            feature_db = _modal_feature_db_values(feature)
            if feature_db is not None:
                vmax_candidates.append(float(feature_db.max().item()))
            vmax = max(vmax_candidates)
            vmin = vmax - MODAL_DYNAMIC_RANGE_DB
            target_png = _plot_modal_feature_png(
                feature,
                duration,
                f"Original modal feature | {feature_path.name}",
                vmin,
                vmax,
            )
            recon_png = _plot_modal_spectrogram_png(
                recon_db,
                recon_freqs,
                duration,
                f"Reconstruction | {row.recon_path.name}",
                vmin,
                vmax,
            )
            target_out.parent.mkdir(parents=True, exist_ok=True)
            target_out.write_bytes(target_png)
        elif target_path is not None:
            target_audio, target_sr = _load_mono(target_path)
            duration = target_audio.shape[0] / max(target_sr, 1)
            target_db, target_freqs, duration = _modal_analysis_spectrogram_db(
                target_audio,
                target_sr,
                duration=duration,
            )
            recon_db, recon_freqs, _ = _modal_analysis_spectrogram_db(
                recon_audio,
                recon_sr,
                duration=duration,
            )
            vmax = max(float(target_db.max().item()), float(recon_db.max().item()))
            vmin = vmax - MODAL_DYNAMIC_RANGE_DB
            target_png = _plot_modal_spectrogram_png(
                target_db,
                target_freqs,
                duration,
                f"Original sample | {target_path.name}",
                vmin,
                vmax,
            )
            recon_png = _plot_modal_spectrogram_png(
                recon_db,
                recon_freqs,
                duration,
                f"Reconstruction | {row.recon_path.name}",
                vmin,
                vmax,
            )
            target_out.parent.mkdir(parents=True, exist_ok=True)
            target_out.write_bytes(target_png)
        else:
            recon_db, recon_freqs, duration = _modal_analysis_spectrogram_db(recon_audio, recon_sr)
            vmax = float(recon_db.max().item())
            vmin = vmax - MODAL_DYNAMIC_RANGE_DB
            recon_png = _plot_modal_spectrogram_png(
                recon_db,
                recon_freqs,
                duration,
                f"Reconstruction | {row.recon_path.name}",
                vmin,
                vmax,
            )

        recon_out.parent.mkdir(parents=True, exist_ok=True)
        recon_out.write_bytes(recon_png)
        return target_png, recon_png

    def _serve_audio(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404, f"Audio not found: {path}")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _pick_rows(self, seed: Optional[int], query: str = "") -> List[SampleRow]:
        if query.strip():
            matches = _filter_rows(self.rows, query)
            return matches[: min(self.n_show, len(matches))]
        rng = random.Random(seed if seed is not None else time.time_ns())
        n = min(self.n_show, len(self.rows))
        return rng.sample(self.rows, n)

    def _render_index(self, seed: Optional[int], query: str = "") -> bytes:
        selected = self._pick_rows(seed, query)
        cards: List[str] = []
        search_value = html.escape(query)
        for i, row in enumerate(selected, start=1):
            sample_name = Path(row.source_filename).name or row.source_filename
            comparison_blocks: List[str] = []
            target_audio_html = "<div class='muted'>Original sample not available.</div>"
            target_wave_html = "<div class='muted'>Original waveform unavailable.</div>"
            target_spec_html = "<div class='muted'>Original spectrogram unavailable.</div>"
            if row.target_path is not None and row.target_path.is_file():
                target_q = urllib.parse.quote(str(row.target_path), safe="")
                target_wave = _b64_png(self._render_png_cached(row.target_path, "wave"))
                target_spec = _b64_png(self._render_png_cached(row.target_path, "spec"))
                target_audio_html = (
                    f"<audio controls preload='none' src='/audio?path={target_q}'></audio>"
                )
                target_wave_html = f"<img class='plot' src='{target_wave}' alt='target_wave'>"
                target_spec_html = f"<img class='plot' src='{target_spec}' alt='target_spectrogram'>"

            comparison_blocks.append(
                f"""
                <section class="comparison-block">
                  <div class="comparison-title">
                    <h4>Original Audio</h4>
                    {target_audio_html}
                  </div>
                  <div class="comparison-viz">
                    <div>
                      <h5>Waveform</h5>
                      {target_wave_html}
                    </div>
                    <div>
                      <h5>Spectrogram</h5>
                      {target_spec_html}
                    </div>
                  </div>
                </section>
                """
            )

            model_rows: List[Tuple[str, Optional[SampleRow]]] = []
            for run in self.runs:
                model_row = _match_row(row, run)
                model_rows.append((run.label, model_row))
                if model_row is None:
                    comparison_blocks.append(
                        "<section class='comparison-block missing'>"
                        "<div class='comparison-title'>"
                        f"<h4>{html.escape(run.label)}</h4>"
                        "<div class='muted'>No matching reconstruction for this sample.</div>"
                        "</div>"
                        "</section>"
                    )
                    continue

                recon_q = urllib.parse.quote(str(model_row.recon_path), safe="")
                recon_wave = _b64_png(self._render_png_cached(model_row.recon_path, "wave"))
                recon_spec = _b64_png(self._render_png_cached(model_row.recon_path, "spec"))
                comparison_blocks.append(
                    f"""
                    <section class="comparison-block">
                      <div class="comparison-title">
                        <h4>{html.escape(run.label)}</h4>
                        <audio controls preload='none' src='/audio?path={recon_q}'></audio>
                      </div>
                      <div class="comparison-viz">
                        <div>
                          <h5>Waveform</h5>
                          <img class='plot' src='{recon_wave}' alt='reconstruction_waveform'>
                        </div>
                        <div>
                          <h5>Spectrogram</h5>
                          <img class='plot' src='{recon_spec}' alt='reconstruction_spectrogram'>
                        </div>
                      </div>
                    </section>
                    """
                )

            card = f"""
            <section class="card">
              <div class="card-header">
                <div>
                  <h3>#{i} {html.escape(sample_name)}</h3>
                  <div class="muted">meta_key={html.escape(row.meta_key)} | length={row.length}</div>
                </div>
                <div class="metrics">{_metric_compare_table_html(model_rows)}</div>
              </div>
              <div class="comparison-stack">
                {''.join(comparison_blocks)}
              </div>
            </section>
            """
            cards.append(card)

        if not cards:
            cards.append(
                "<section class='card'><h3>No matching samples</h3>"
                "<p class='muted'>Try a shorter filename fragment, pack name, or meta key.</p></section>"
            )

        status_text = (
            f"Showing {len(selected)} search matches"
            if query.strip()
            else f"Showing {len(selected)} random samples"
        )
        search_json = _rows_search_json(self.rows)

        html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DrumBlender Audition Viewer</title>
  <style>
    :root {{
      --bg: #f3f5f7;
      --card: #ffffff;
      --text: #1a1e24;
      --muted: #5f6b7a;
      --accent: #0f766e;
      --border: #d8dee6;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 0% 0%, #e8f5ef 0, #f3f5f7 35%);
    }}
    header {{
      position: sticky;
      top: 0;
      backdrop-filter: blur(6px);
      background: rgba(243,245,247,0.88);
      border-bottom: 1px solid var(--border);
      padding: 14px 20px;
      z-index: 10;
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(320px, 560px) auto;
      gap: 14px;
      align-items: start;
    }}
    h1 {{ margin: 0; font-size: 20px; }}
    .controls {{ display: flex; gap: 10px; align-items: center; justify-content: flex-end; }}
    .search-panel {{ min-width: 0; }}
    .search-row {{ display: flex; gap: 8px; align-items: center; }}
    .search-row label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    #sampleSearch {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 9px;
      padding: 8px 10px;
      font-size: 14px;
      background: #fff;
      color: var(--text);
    }}
    .search-results {{
      display: none;
      margin-top: 8px;
      max-height: 230px;
      overflow-y: auto;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fff;
      box-shadow: 0 8px 24px rgba(16,24,40,0.10);
    }}
    .search-panel.has-results .search-results {{ display: block; }}
    .search-result {{
      display: block;
      width: 100%;
      border: 0;
      border-bottom: 1px solid #edf1f5;
      border-radius: 0;
      background: #fff;
      color: var(--text);
      padding: 8px 10px;
      text-align: left;
      cursor: pointer;
    }}
    .search-result:hover, .search-result:focus {{ background: #edf7f4; outline: none; }}
    .search-result strong {{ display: block; font-size: 12px; overflow-wrap: anywhere; }}
    .search-result span {{ display: block; margin-top: 2px; color: var(--muted); font-size: 11px; }}
    button {{
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 14px;
      cursor: pointer;
    }}
    .wrap {{ padding: 16px; max-width: 1400px; margin: 0 auto; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 14px;
      box-shadow: 0 2px 10px rgba(16,24,40,0.04);
    }}
    .card-header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 14px;
      align-items: start;
    }}
    .card h3 {{ margin: 0 0 4px 0; font-size: 18px; }}
    .card h4 {{ margin: 0 0 8px 0; font-size: 14px; color: #1f2937; }}
    .card h5 {{ margin: 10px 0 6px 0; font-size: 12px; color: #374151; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .metrics {{ min-width: 240px; }}
    .metric-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
    }}
    .metric-table td, .metric-table th {{
      border-bottom: 1px solid #edf1f5;
      padding: 4px 6px;
      vertical-align: top;
      text-align: right;
    }}
    .metric-table td:first-child, .metric-table th:first-child {{
      text-align: left;
    }}
    .metric-compare th {{
      color: var(--muted);
      font-weight: 700;
    }}
    .comparison-stack {{
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }}
    .comparison-block {{
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fbfcfd;
      padding: 12px;
    }}
    .comparison-block.missing {{
      display: flex;
      min-height: 140px;
      flex-direction: column;
      justify-content: center;
      background: #fff8f1;
    }}
    .comparison-title {{
      display: flex;
      gap: 8px;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 8px;
    }}
    .comparison-title h4 {{ margin: 0; white-space: nowrap; }}
    .comparison-title audio {{ max-width: min(520px, 65%); }}
    .comparison-viz {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }}
    audio {{ width: 100%; }}
    .plot {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
    }}
    @media (max-width: 900px) {{
      header, .card-header, .comparison-viz {{
        grid-template-columns: 1fr;
      }}
      .comparison-title {{
        display: grid;
        gap: 8px;
      }}
      .comparison-title audio {{ max-width: 100%; }}
      .controls {{ justify-content: flex-start; flex-wrap: wrap; }}
      .metrics {{ min-width: 0; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>DrumBlender Audition Viewer</h1>
    <div class="search-panel" id="searchPanel">
      <div class="search-row">
        <label for="sampleSearch">Search sample</label>
        <input id="sampleSearch" type="search" value="{search_value}" placeholder="filename, pack, meta key..." autocomplete="off" />
      </div>
      <div class="search-results" id="searchResults"></div>
    </div>
    <div class="controls">
      <div class="muted">{html.escape(status_text)}</div>
      <button onclick="refreshSamples()">Refresh Random Samples</button>
      <button onclick="clearSearch()">Clear Search</button>
    </div>
  </header>
  <main class="wrap">
    {''.join(cards)}
  </main>
  <script>
    const SAMPLES = {search_json};

    function refreshSamples() {{
      const u = new URL(window.location.href);
      u.searchParams.set("seed", String(Date.now()));
      u.searchParams.delete("q");
      window.location.href = u.toString();
    }}

    function clearSearch() {{
      const u = new URL(window.location.href);
      u.searchParams.delete("q");
      u.searchParams.delete("seed");
      window.location.href = u.toString();
    }}

    function applySearch(value) {{
      const q = value.trim();
      const u = new URL(window.location.href);
      u.searchParams.delete("seed");
      if (q) {{
        u.searchParams.set("q", q);
      }} else {{
        u.searchParams.delete("q");
      }}
      window.location.href = u.toString();
    }}

    function renderSearchResults() {{
      const panel = document.getElementById("searchPanel");
      const input = document.getElementById("sampleSearch");
      const results = document.getElementById("searchResults");
      const q = input.value.trim().toLowerCase();
      results.replaceChildren();
      if (!q) {{
        panel.classList.remove("has-results");
        return;
      }}
      const terms = q.split(/\\s+/).filter(Boolean);
      const matches = SAMPLES
        .filter((sample) => terms.every((term) => sample.haystack.includes(term)))
        .slice(0, 80);
      if (!matches.length) {{
        const empty = document.createElement("div");
        empty.className = "search-result";
        empty.textContent = "No matching samples";
        results.append(empty);
        panel.classList.add("has-results");
        return;
      }}
      for (const sample of matches) {{
        const button = document.createElement("button");
        button.className = "search-result";
        button.type = "button";
        const title = document.createElement("strong");
        title.textContent = sample.source_display;
        const meta = document.createElement("span");
        meta.textContent = `index=${{sample.index}} | meta_key=${{sample.meta_key}}`;
        button.append(title, meta);
        button.addEventListener("click", () => applySearch(sample.source_display));
        results.append(button);
      }}
      panel.classList.add("has-results");
    }}

    const searchInput = document.getElementById("sampleSearch");
    searchInput.addEventListener("input", renderSearchResults);
    searchInput.addEventListener("focus", renderSearchResults);
    searchInput.addEventListener("keydown", (event) => {{
      if (event.key === "Enter") {{
        event.preventDefault();
        applySearch(searchInput.value);
      }}
    }});
  </script>
</body>
</html>
"""
        return html_doc.encode("utf-8")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        route = parsed.path

        if route == "/" or route == "/index.html":
            seed = None
            if "seed" in qs and len(qs["seed"]) > 0:
                try:
                    seed = int(qs["seed"][0])
                except Exception:
                    seed = None
            query = qs.get("q", [""])[0].strip()
            page = self._render_index(seed, query)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        if route == "/audio":
            p = qs.get("path", [""])[0]
            if p == "":
                self.send_error(400, "Missing path")
                return
            path = Path(urllib.parse.unquote(p))
            self._serve_audio(path)
            return

        self.send_error(404, "Not Found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_dir",
        type=str,
        nargs="?",
        default=DEFAULT_RESULT_DIR,
        help="Result folder exported by export_recon_wavs.py, or a top-level run directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Legacy modal feature dataset override. Current viewer renders audio spectrograms from WAVs.",
    )
    parser.add_argument(
        "--compare-result-dir",
        action="append",
        default=None,
        help=(
            "Additional result folder to compare against the primary result. "
            "Default is ../results/run_NOISEDAC_20260412_231956."
        ),
    )
    parser.add_argument(
        "--compare-dataset-dir",
        action="append",
        default=None,
        help="Legacy modal feature dataset override for each --compare-result-dir.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help=(
            "Disable the default Ref. NOISEDAC comparison and show only the primary result."
        ),
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--num-samples", type=int, default=10, help="Samples per page")
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Open browser automatically (default: on)",
    )
    parser.add_argument(
        "--no-open",
        dest="open_browser",
        action="store_false",
        help="Do not open browser automatically",
    )
    parser.set_defaults(open_browser=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs: List[RunView] = [_build_run_view(args.result_dir, args.dataset_dir)]
    if args.compare_result_dir is not None:
        compare_inputs = list(args.compare_result_dir)
    elif args.single:
        compare_inputs = []
    else:
        compare_inputs = [DEFAULT_COMPARE_RESULT_DIR]

    compare_dataset_inputs = list(args.compare_dataset_dir or [])
    for idx, compare_input in enumerate(compare_inputs):
        dataset_input = compare_dataset_inputs[idx] if idx < len(compare_dataset_inputs) else None
        runs.append(_build_run_view(compare_input, dataset_input))

    primary_run = runs[0]
    print(f"[OK] Loaded {len(primary_run.rows)} samples from: {primary_run.result_dir}")
    for run in runs:
        print(f"[OK] {run.label}: result={run.result_dir}")
        if run.dataset_dir is not None:
            print(f"[OK] {run.label}: modal features={run.dataset_dir}")

    AuditionHandler.rows = primary_run.rows
    AuditionHandler.runs = runs
    AuditionHandler.n_show = max(1, int(args.num_samples))
    AuditionHandler.cache_dir = primary_run.result_dir / ".viz_cache"
    AuditionHandler.cache_dir.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, int(args.port)), AuditionHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[OK] Serving audition viewer at: {url}")

    if args.open_browser:
        # ### HIGHLIGHT: Browser open is non-fatal for headless environments.
        def _open_later() -> None:
            try:
                time.sleep(0.4)
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open_later, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
