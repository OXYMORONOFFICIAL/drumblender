from __future__ import annotations

import argparse
import csv
import inspect
import random
import sys
from pathlib import Path
from typing import Dict, Tuple, Type

import matplotlib

if "--no_show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import torch
import torchaudio

from drumblender.synths.modal import modal_synth
from drumblender.utils.modal_analysis import CQTModalAnalysis as AmpZeroModalAnalysis
from drumblender.utils.modal_analysis_NEW import CQTModalAnalysis as NewModalAnalysis
from drumblender.utils.modal_analysis_OLD import CQTModalAnalysis as OldModalAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original audio spectrogram against modal-only "
            "reconstructions from OLD, NEW, and current amp-zero-filtered "
            "modal analysis."
        )
    )
    parser.add_argument("--samples_dir", type=str, default="../samples/processed")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="../results/run_NOISEDAC_20260412_231956/all/05_all_parallel_all",
        help="Optional exported reconstruction bundle with manifest.csv.",
    )
    parser.add_argument(
        "--new_result_dir",
        type=str,
        default="../results/run_NOISEDAC_20260414_111726/all/05_all_parallel_all",
        help="Optional new-way exported reconstruction bundle with manifest.csv.",
    )
    parser.add_argument("--sample", type=str, default="")
    parser.add_argument(
        "--random",
        dest="random_sample",
        action="store_true",
        help="Pick a random wav from samples_dir when --sample is not set.",
    )
    parser.add_argument(
        "--no_random",
        dest="random_sample",
        action="store_false",
        help="Pick the first sorted wav from samples_dir when --sample is not set.",
    )
    parser.set_defaults(random_sample=True)
    parser.add_argument("--seed", type=int, default=0, help="Random sample seed.")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional PNG path. If empty, the plot is only shown interactively.",
    )
    parser.add_argument(
        "--no_show",
        dest="show",
        action="store_false",
        help="Do not call plt.show(); useful when only saving with --out.",
    )
    parser.set_defaults(show=True)

    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--n_bins", type=int, default=240)
    parser.add_argument("--bins_per_octave", type=int, default=24)
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--threshold_db", type=float, default=-80.0)
    parser.add_argument("--diff_threshold", type=float, default=5.0)
    parser.add_argument("--p_kernel", type=int, default=31)

    parser.add_argument("--max_gap", type=int, default=2)
    parser.add_argument("--min_active_frames", type=int, default=0)
    parser.add_argument("--min_streak", type=int, default=0)
    parser.add_argument("--min_active_ratio", type=float, default=0.25)
    parser.add_argument("--min_track_energy", type=float, default=0.0)

    parser.add_argument(
        "--min_samples",
        type=int,
        default=131072,
        help="Right-pad shorter samples before CQT analysis to avoid reflect-pad failures.",
    )
    parser.add_argument("--dynamic_range_db", type=float, default=100.0)
    parser.add_argument("--fmax", type=float, default=22050.0)
    parser.add_argument("--log_freq_min", type=float, default=20.0)
    parser.add_argument("--track_line_width", type=float, default=1.2)
    return parser.parse_args()


def localize_result_path(path_str: str, result_dir: Path) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    text = str(path_str).replace("\\", "/")
    marker = "/05_all_parallel_all/"
    if marker in text:
        rel = text.split(marker, 1)[1]
        candidate = result_dir / rel
        if candidate.exists():
            return candidate

    marker = "/run_NOISEDAC_20260412_231956/all/05_all_parallel_all/"
    if marker in text:
        rel = text.split(marker, 1)[1]
        candidate = result_dir / rel
        if candidate.exists():
            return candidate

    return path


def load_result_rows(result_dir: Path) -> list[dict[str, str]]:
    manifest = result_dir / "manifest.csv"
    if not manifest.exists():
        return []
    with manifest.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row_sample_path(row: dict[str, str], samples_dir: Path) -> Path:
    return samples_dir / row["source_filename"]


def row_recon_path(row: dict[str, str], result_dir: Path) -> Path:
    return localize_result_path(row["recon_path"], result_dir)


def find_row_by_sample(
    sample: Path,
    rows: list[dict[str, str]],
    samples_dir: Path,
) -> dict[str, str] | None:
    for row in rows:
        if row_sample_path(row, samples_dir).resolve() == sample:
            return row
    return None


def find_recon_for_source(
    source_filename: str,
    result_dir: Path,
) -> Path | None:
    for row in load_result_rows(result_dir):
        if row["source_filename"] == source_filename:
            recon = row_recon_path(row, result_dir)
            return recon if recon.exists() else None
    return None


def find_sample(args: argparse.Namespace) -> Tuple[Path, Path | None, Path | None]:
    samples_dir = Path(args.samples_dir)
    result_dir = Path(args.result_dir)
    new_result_dir = Path(args.new_result_dir)
    if args.sample:
        sample = Path(args.sample)
        sample = sample if sample.is_absolute() else sample.resolve()
        row = find_row_by_sample(sample, load_result_rows(result_dir), samples_dir)
        if row is None:
            return sample, None, None

        recon = row_recon_path(row, result_dir)
        new_recon = find_recon_for_source(row["source_filename"], new_result_dir)
        return sample, recon if recon.exists() else None, new_recon

    result_rows = [
        row
        for row in load_result_rows(result_dir)
        if row_sample_path(row, samples_dir).exists()
        and row_recon_path(row, result_dir).exists()
    ]
    if result_rows:
        result_rows.sort(key=lambda row: row["source_filename"])
        row = random.Random(args.seed).choice(result_rows) if args.random_sample else result_rows[0]
        return (
            row_sample_path(row, samples_dir),
            row_recon_path(row, result_dir),
            find_recon_for_source(row["source_filename"], new_result_dir),
        )

    root = samples_dir
    wavs = []
    flat_audio_wavs = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        rel_parts = path.relative_to(root).parts
        if "modal_features" in rel_parts:
            if rel_parts[-3:-1] == ("processed_modal_flat", "audio"):
                flat_audio_wavs.append(path)
            continue
        wavs.append(path)
    if not wavs:
        wavs = flat_audio_wavs
    wavs.sort()
    if not wavs:
        raise FileNotFoundError(f"No .wav files found under {root}")
    if args.random_sample:
        return random.Random(args.seed).choice(wavs), None, None
    return wavs[0], None, None


def load_audio(path: Path, sample_rate: int, min_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    analysis_waveform = waveform
    if analysis_waveform.shape[-1] < min_samples:
        analysis_waveform = torch.nn.functional.pad(
            analysis_waveform, (0, min_samples - analysis_waveform.shape[-1])
        )
    return waveform, analysis_waveform


def modal_kwargs(
    cls: Type,
    args: argparse.Namespace,
) -> Dict:
    values = {
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "fmin": args.fmin,
        "n_bins": args.n_bins,
        "bins_per_octave": args.bins_per_octave,
        "min_length": args.min_length,
        "num_modes": args.num_modes,
        "threshold": args.threshold_db,
        "diff_threshold": args.diff_threshold,
        "p_kernel": args.p_kernel,
        "max_gap": args.max_gap,
        "min_active_frames": args.min_active_frames or None,
        "min_streak": args.min_streak or None,
        "min_active_ratio": args.min_active_ratio,
        "min_track_energy": args.min_track_energy,
    }
    params = inspect.signature(cls.__init__).parameters
    return {key: value for key, value in values.items() if key in params}


@torch.no_grad()
def modal_tracks(
    cls: Type,
    waveform: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    modal = cls(**modal_kwargs(cls, args))
    try:
        freqs, amps, phases = modal(waveform)
    except Exception as exc:
        empty = torch.zeros((0, 0))
        return empty, empty, f"failed: {type(exc).__name__}"

    if freqs.numel() == 0 or freqs.shape[1] == 0:
        empty = torch.zeros((0, 0))
        return empty, empty, "0 modes"

    finite = torch.isfinite(freqs).all() and torch.isfinite(amps).all() and torch.isfinite(phases).all()
    if not finite:
        freqs = torch.nan_to_num(freqs, nan=0.0, posinf=0.0, neginf=0.0)
        amps = torch.nan_to_num(amps, nan=0.0, posinf=0.0, neginf=0.0)

    freqs = freqs[0].cpu()
    amps = amps[0].cpu()
    active_modes = int((amps > 0).any(dim=1).sum())
    status = f"{active_modes} modes"
    if not finite:
        status += ", non-finite sanitized"
    return freqs, amps, status


def modal_analysis_spectrogram_db(
    waveform: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    modal = AmpZeroModalAnalysis(**modal_kwargs(AmpZeroModalAnalysis, args))
    mag = modal.spectrogram(waveform, complex=False).squeeze(0).clamp_min(1e-8)
    db = 20.0 * torch.log10(mag)
    freqs_hz = torch.as_tensor(modal.frequencies(), dtype=db.dtype, device=db.device)
    keep = freqs_hz <= min(args.fmax, args.sample_rate / 2.0)
    return db[keep].cpu(), freqs_hz[keep].cpu()


@torch.no_grad()
def modal_branch_spectrogram_db(
    cls: Type,
    waveform: torch.Tensor,
    num_samples: int,
    args: argparse.Namespace,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor] | None, str]:
    modal = cls(**modal_kwargs(cls, args))
    try:
        freqs_hz, amps, phases = modal(waveform)
    except Exception as exc:
        return None, f"failed: {type(exc).__name__}"

    if freqs_hz.numel() == 0 or freqs_hz.shape[1] == 0:
        return None, "0 modes"

    finite = (
        torch.isfinite(freqs_hz).all()
        and torch.isfinite(amps).all()
        and torch.isfinite(phases).all()
    )
    if not finite:
        freqs_hz = torch.nan_to_num(freqs_hz, nan=0.0, posinf=0.0, neginf=0.0)
        amps = torch.nan_to_num(amps, nan=0.0, posinf=0.0, neginf=0.0)
        phases = torch.nan_to_num(phases, nan=0.0, posinf=0.0, neginf=0.0)

    active_modes = int((amps[0] > 0).any(dim=1).sum())
    freqs = 2.0 * torch.pi * freqs_hz / args.sample_rate
    synth = modal_synth(freqs, amps, num_samples, phases)
    if synth.shape[-1] < args.min_samples:
        synth = torch.nn.functional.pad(synth, (0, args.min_samples - synth.shape[-1]))

    status = f"{active_modes} modes"
    if not finite:
        status += ", non-finite sanitized"
    return modal_analysis_spectrogram_db(synth, args), status


def frequency_edges_hz(freqs_hz: torch.Tensor) -> torch.Tensor:
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


def modal_line_collection(
    freqs_hz: torch.Tensor,
    amps: torch.Tensor,
    duration: float,
    args: argparse.Namespace,
    lower_hz: float,
    upper_hz: float,
    vmin: float,
    vmax: float,
) -> LineCollection | None:
    if freqs_hz.numel() == 0 or amps.numel() == 0 or freqs_hz.shape[1] < 2:
        return None

    # ModalSynth interpolates the feature-frame axis directly to num_samples.
    # Map frames across the output duration the same way for visualization.
    frame_times = torch.linspace(0.0, duration, freqs_hz.shape[1])
    t0 = frame_times[:-1].unsqueeze(0).expand(freqs_hz.shape[0], -1)
    t1 = frame_times[1:].unsqueeze(0).expand(freqs_hz.shape[0], -1)
    f0 = freqs_hz[:, :-1]
    f1 = freqs_hz[:, 1:]
    a0 = amps[:, :-1]
    a1 = amps[:, 1:]

    # Presentation view: hide terminal fade-to-zero segments so the plot focuses
    # on active tracks rather than visually harsh endpoint padding.
    mask = (
        (a0 > 0)
        & (a1 > 0)
        & (f0 > 0)
        & (f1 > 0)
        & (f0 >= lower_hz)
        & (f0 <= upper_hz)
        & (f1 >= lower_hz)
        & (f1 <= upper_hz)
        & (t0 <= duration)
    )
    if not bool(mask.any()):
        return None

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
        linewidths=args.track_line_width,
        alpha=0.95,
        capstyle="round",
        joinstyle="round",
    )
    collection.set_array(segment_db.numpy())
    return collection


def plot_specs(
    original_spec: Tuple[torch.Tensor, torch.Tensor],
    modal_branch_specs: Dict[str, Tuple[Tuple[torch.Tensor, torch.Tensor] | None, str]],
    recon_spec: Tuple[torch.Tensor, torch.Tensor] | None,
    new_recon_spec: Tuple[torch.Tensor, torch.Tensor] | None,
    tracks: Dict[str, Tuple[torch.Tensor, torch.Tensor, str]],
    out_path: Path | None,
    sample_path: Path,
    recon_path: Path | None,
    new_recon_path: Path | None,
    duration: float,
    args: argparse.Namespace,
) -> None:
    original_db, original_freqs_hz = original_spec
    amp_db_values = []
    for freqs_hz, amps, _ in tracks.values():
        mask = (amps > 0) & (freqs_hz > 0)
        if bool(mask.any()):
            amp_db_values.append(20.0 * torch.log10(amps[mask].clamp_min(1e-8)))

    vmax_candidates = [float(original_db.max())]
    for spec, _ in modal_branch_specs.values():
        if spec is not None:
            vmax_candidates.append(float(spec[0].max()))
    if recon_spec is not None:
        vmax_candidates.append(float(recon_spec[0].max()))
    if new_recon_spec is not None:
        vmax_candidates.append(float(new_recon_spec[0].max()))
    if amp_db_values:
        vmax_candidates.extend(float(v.max()) for v in amp_db_values)
    vmax = max(vmax_candidates)
    vmin = vmax - args.dynamic_range_db
    fmax = min(args.fmax, args.sample_rate / 2.0)
    lower_hz = max(args.log_freq_min, 1e-6)
    upper_hz = fmax
    ticks_hz = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ticks_hz = [tick for tick in ticks_hz if lower_hz <= tick <= upper_hz]

    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)

    original_ax = axes.flat[0]
    original_freq_edges = frequency_edges_hz(original_freqs_hz)
    original_time_edges = torch.linspace(0.0, duration, original_db.shape[1] + 1)
    image = original_ax.pcolormesh(
        original_time_edges.numpy(),
        original_freq_edges.numpy(),
        original_db.numpy(),
        shading="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    original_ax.set_title("Original sample")

    for ax, (title, (freqs_hz, amps, status)) in zip(axes.flat[1:], tracks.items()):
        ax.set_facecolor("black")
        collection = modal_line_collection(
            freqs_hz=freqs_hz,
            amps=amps,
            duration=duration,
            args=args,
            lower_hz=lower_hz,
            upper_hz=upper_hz,
            vmin=vmin,
            vmax=vmax,
        )
        if collection is None:
            ax.text(0.5, 0.5, status, ha="center", va="center", transform=ax.transAxes)
        else:
            image = collection
            ax.add_collection(collection)
        ax.set_title(f"{title} ({status})")

    spec_plots = [
        (
            axes.flat[4],
            modal_branch_specs["Original modal branch synthesis"][0],
            None,
            f"Original modal branch synthesis ({modal_branch_specs['Original modal branch synthesis'][1]})",
        ),
        (
            axes.flat[5],
            modal_branch_specs["NEW modal branch synthesis"][0],
            None,
            f"NEW modal branch synthesis ({modal_branch_specs['NEW modal branch synthesis'][1]})",
        ),
        (axes.flat[6], recon_spec, recon_path, "Old-way exported reconstruction"),
        (axes.flat[7], new_recon_spec, new_recon_path, "New-way exported reconstruction"),
    ]
    for ax, spec, path, title in spec_plots:
        if spec is None:
            ax.set_axis_off()
            continue
        recon_db, recon_freqs_hz = spec
        recon_freq_edges = frequency_edges_hz(recon_freqs_hz)
        recon_time_edges = torch.linspace(0.0, duration, recon_db.shape[1] + 1)
        image = ax.pcolormesh(
            recon_time_edges.numpy(),
            recon_freq_edges.numpy(),
            recon_db.numpy(),
            shading="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        if path is not None:
            title += f"\n{path.name}"
        ax.set_title(title)

    for ax in axes.flat:
        if not ax.axison:
            continue
        ax.set_yscale("log")
        ax.set_ylim(lower_hz, upper_hz)
        ax.set_xlim(0.0, duration)
        if ticks_hz:
            ax.set_yticks(ticks_hz)
            ax.set_yticklabels([str(tick) for tick in ticks_hz])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    fig.suptitle(str(sample_path), fontsize=11)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="dB")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
    if args.show:
        plt.show()
    else:
        plt.close(fig)

def main() -> None:
    args = parse_args()
    sample_path, recon_path, new_recon_path = find_sample(args)
    waveform, analysis_waveform = load_audio(
        sample_path, args.sample_rate, args.min_samples
    )
    target_num_samples = waveform.shape[-1]
    recon_spec = None
    if recon_path is not None and recon_path.exists():
        recon_waveform, recon_analysis_waveform = load_audio(
            recon_path, args.sample_rate, args.min_samples
        )
        recon_spec = modal_analysis_spectrogram_db(recon_analysis_waveform, args)
    new_recon_spec = None
    if new_recon_path is not None and new_recon_path.exists():
        new_recon_waveform, new_recon_analysis_waveform = load_audio(
            new_recon_path, args.sample_rate, args.min_samples
        )
        new_recon_spec = modal_analysis_spectrogram_db(
            new_recon_analysis_waveform, args
        )

    tracks = {
        "Original modal analysis": modal_tracks(
            OldModalAnalysis, analysis_waveform, args
        ),
        "Modal analysis NEW": modal_tracks(
            NewModalAnalysis, analysis_waveform, args
        ),
        "Amp-zero filtered modal analysis": modal_tracks(
            AmpZeroModalAnalysis, analysis_waveform, args
        ),
    }

    original_spec = modal_analysis_spectrogram_db(analysis_waveform, args)

    out_path = Path(args.out) if args.out else None
    plot_specs(
        original_spec,
        {
            "Original modal branch synthesis": modal_branch_spectrogram_db(
                OldModalAnalysis, analysis_waveform, target_num_samples, args
            ),
            "NEW modal branch synthesis": modal_branch_spectrogram_db(
                NewModalAnalysis, analysis_waveform, target_num_samples, args
            ),
        },
        recon_spec,
        new_recon_spec,
        tracks,
        out_path,
        sample_path,
        recon_path,
        new_recon_path,
        target_num_samples / args.sample_rate,
        args,
    )
    print(f"sample: {sample_path}")
    if recon_path is not None:
        print(f"recon: {recon_path}")
    if new_recon_path is not None:
        print(f"new_recon: {new_recon_path}")
    if out_path is not None:
        print(f"output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
