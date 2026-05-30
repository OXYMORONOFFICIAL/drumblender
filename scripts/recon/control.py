#!/usr/bin/env python3
"""
Local transient-latent control audition server.

The controls are descriptor-free and operate only on the transient branch latent:

- pca: major latent-distribution axes.
- sensitivity: axes where tiny transient-latent perturbations most affect the
  rendered waveform, estimated with Hutchinson Jacobian probes.

Knob values are fixed to -1, -0.5, 0, +0.5, +1. Zero is the encoded sample;
one unit means one latent projection standard deviation along the selected axis.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import random
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from urllib.parse import parse_qs
from urllib.parse import urlparse

DEFAULT_RUN_DIR = "../results/run_NOISEDAC_20260412_231956"
DEFAULT_DATA_DIR = "../datasets/modal_features/processed_modal_flat"
DEFAULT_DATA_DIR_0412 = "../datasets/modal_features/processed_modal_flat"
DEFAULT_DATA_DIR_0414 = "../datasets/modal_features/processed_modal_flat"
DEFAULT_AUDIO_DIR = "../samples/processed"
DEFAULT_SEED = 20260218
DEFAULT_SR = 48000

torch = None
torchaudio = None
yaml = None
AudioWithParametersDataset = None
load_model = None


def _load_runtime() -> None:
    global AudioWithParametersDataset, load_model, torch, torchaudio, yaml

    import torch as torch_module
    import torchaudio as torchaudio_module
    import yaml as yaml_module

    from drumblender.data.audio import AudioWithParametersDataset as DatasetClass
    from drumblender.utils.model import load_model as load_model_fn

    torch = torch_module
    torchaudio = torchaudio_module
    yaml = yaml_module
    AudioWithParametersDataset = DatasetClass
    load_model = load_model_fn


@dataclass
class LatentAxis:
    name: str
    method: str
    axis_flat: Any
    score: float
    note: str


@dataclass
class SampleState:
    sample_index: int
    sample_key: str
    sample_filename: str
    length: int
    target_waveform: Any
    modal_params: Any
    noise_lat_base: Optional[Any]
    transient_lat_base: Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _latest(paths: Sequence[Path]) -> Optional[Path]:
    files = [p for p in paths if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _device(value: str):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _localize_path(path_str: Optional[str], root: Path) -> Optional[str]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return str(p.resolve())

    text = str(path_str).replace("\\", "/")
    if "/workspace/drumblender/" in text:
        rel = text.split("/workspace/drumblender/", 1)[1]
        q = (root / rel).resolve()
        if q.exists():
            return str(q)
    if text.startswith("/workspace/"):
        rel = text.split("/workspace/", 1)[1]
        q = (root.parent / rel).resolve()
        if q.exists():
            return str(q)
    if not p.is_absolute():
        q = (root / p).resolve()
        if q.exists():
            return str(q)
    return None


def _default_data_dir_for_run(run_dir: Path) -> str:
    text = str(run_dir).replace("\\", "/")
    if "20260414" in text:
        return DEFAULT_DATA_DIR_0414
    if "20260412" in text:
        return DEFAULT_DATA_DIR_0412
    return DEFAULT_DATA_DIR


def _bundle_dirs(run_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    if (
        (run_dir / "evaluation" / "summary.json").is_file()
        or (run_dir / "manifest.csv").is_file()
        or (run_dir / "configs").is_dir()
    ):
        candidates.append(run_dir)
    candidates.extend(sorted((run_dir / "all").glob("*")))
    candidates.extend(sorted((run_dir / "per_pack").glob("*")))

    out: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.is_dir():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _read_context(run_dir: Path) -> Dict[str, Any]:
    candidates = list(run_dir.glob("run-context-*.json"))
    candidates += list((run_dir / "training_context").glob("run-context-*.json"))
    candidates += list((run_dir / "training_context" / "run_context").glob("run-context-*.json"))
    candidates += list(run_dir.glob("**/training_context/run_context/run-context-*.json"))
    candidates += list(run_dir.glob("**/run-context-*.json"))
    path = _latest(candidates)
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_config(run_dir: Path, explicit: Optional[str], ctx: Dict[str, Any], root: Path) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--config not found: {path}")
        return path

    localized = _localize_path(ctx.get("cfg"), root)
    if localized:
        return Path(localized)

    fallback = root / "cfg" / "05_all_parallel.yaml"
    if fallback.is_file():
        return fallback.resolve()

    candidates: List[Path] = []
    for bundle_dir in _bundle_dirs(run_dir):
        candidates.extend((bundle_dir / "configs").glob("05_all_parallel.yaml"))
        candidates.extend((bundle_dir / "configs").glob("*.yaml"))
        candidates.extend((bundle_dir / "configs").glob("*.yml"))
    path = _latest(candidates)
    if path is None:
        raise FileNotFoundError("Could not infer config.")
    return path.resolve()


def _infer_ckpt(run_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"--ckpt not found: {path}")

    candidates: List[Path] = []
    for bundle_dir in _bundle_dirs(run_dir):
        candidates.extend((bundle_dir / "checkpoints").glob("*.ckpt"))
        candidates.extend(bundle_dir.glob("*.ckpt"))
    path = _latest(candidates)
    if path is not None:
        return path.resolve()

    path = _latest(list(run_dir.rglob("*.ckpt")))
    if path is None:
        raise FileNotFoundError("Could not infer checkpoint in run_dir.")
    return path.resolve()


def _backbone_from_autoencoder_path(value: object) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("\\", "/").lower()
    for name in (
        "dac_len_lstm_len",
        "dac_lstm_len",
        "dac_len",
        "dac_lstm",
        "wavtokenizer",
        "spectrostream",
        "bscodec",
        "apcodec",
        "dac",
        "soundstream",
    ):
        if name in text:
            return name
    return None


def _infer_encoder_backbone_from_bundle(run_dir: Path, kind: str) -> Optional[str]:
    key = f"{kind}_autoencoder"
    candidates: List[Path] = []
    for bundle_dir in _bundle_dirs(run_dir):
        candidates.append(bundle_dir / "configs" / "resolved_export_config.yaml")
        candidates.append(bundle_dir / "configs" / "05_all_parallel.yaml")

    for cfg_path in candidates:
        if not cfg_path.is_file():
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        init_args = cfg.get("model", {}).get("init_args", {})
        inferred = _backbone_from_autoencoder_path(init_args.get(key))
        if inferred is not None:
            return inferred
    return None


def _resolve_encoder_cfg(
    cfg_dir: Path,
    kind: str,
    backbone: str,
    explicit: Optional[str],
    root: Path,
) -> Optional[str]:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = (root / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{kind} encoder cfg not found: {path}")
        return str(path)
    if backbone == "soundstream":
        return None
    path = (cfg_dir / "upgrades" / "encoders" / f"{kind}_{backbone}_style.yaml").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing encoder cfg: {path}")
    return str(path)


def _build_resolved_config(
    base_cfg: Path,
    loss_cfg: Optional[str],
    noise_backbone: str,
    transient_backbone: str,
    noise_cfg: Optional[str],
    transient_cfg: Optional[str],
    root: Path,
) -> tuple[Path, bool]:
    cfg_dir = base_cfg.parent
    if (
        not any([loss_cfg, noise_cfg, transient_cfg])
        and noise_backbone == "soundstream"
        and transient_backbone == "soundstream"
    ):
        return base_cfg, False

    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    init_args = cfg.setdefault("model", {}).setdefault("init_args", {})
    if loss_cfg:
        init_args["loss_fn"] = str(Path(loss_cfg).resolve())

    noise_resolved = _resolve_encoder_cfg(cfg_dir, "noise", noise_backbone, noise_cfg, root)
    if noise_resolved:
        init_args["noise_autoencoder"] = noise_resolved
        init_args["noise_autoencoder_accepts_audio"] = True

    transient_resolved = _resolve_encoder_cfg(
        cfg_dir,
        "transient",
        transient_backbone,
        transient_cfg,
        root,
    )
    if transient_resolved:
        init_args["transient_autoencoder"] = transient_resolved
        init_args["transient_autoencoder_accepts_audio"] = True

    tmp = cfg_dir / f".control_resolved_{time.time_ns()}.yaml"
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return tmp, True


def _infer_backbone(args: argparse.Namespace, ctx: Dict[str, Any], run_dir: Path, kind: str) -> str:
    attr = f"{kind}_encoder_backbone"
    return str(
        getattr(args, attr)
        or ctx.get(attr)
        or _infer_encoder_backbone_from_bundle(run_dir, kind)
        or "soundstream"
    )


def _resolve_model_and_data(args: argparse.Namespace):
    root = _repo_root()
    run_dir = Path(args.run_dir).expanduser().resolve()
    ctx = _read_context(run_dir)
    cfg = _infer_config(run_dir, args.config, ctx, root)
    ckpt = _infer_ckpt(run_dir, args.ckpt)
    data_dir = Path(
        args.data_dir
        or _localize_path(ctx.get("data_dir"), root)
        or root / _default_data_dir_for_run(run_dir)
    ).expanduser().resolve()
    audio_dir = Path(
        args.audio_dir
        or _localize_path(ctx.get("audio_dir"), root)
        or root / DEFAULT_AUDIO_DIR
    ).expanduser().resolve()

    noise_backbone = _infer_backbone(args, ctx, run_dir, "noise")
    transient_backbone = _infer_backbone(args, ctx, run_dir, "transient")
    noise_cfg = args.noise_encoder_cfg or _localize_path(ctx.get("noise_encoder_cfg"), root)
    transient_cfg = args.transient_encoder_cfg or _localize_path(
        ctx.get("transient_encoder_cfg"),
        root,
    )
    loss_cfg = args.loss_cfg or _localize_path(
        ctx.get("loss_cfg") if str(ctx.get("loss_upgrade", "")).lower() == "on" else None,
        root,
    )

    resolved_cfg, is_tmp = _build_resolved_config(
        base_cfg=cfg,
        loss_cfg=loss_cfg,
        noise_backbone=noise_backbone,
        transient_backbone=transient_backbone,
        noise_cfg=noise_cfg,
        transient_cfg=transient_cfg,
        root=root,
    )
    try:
        model, _ = load_model(str(resolved_cfg), str(ckpt), include_data=False)
    finally:
        if is_tmp and resolved_cfg.exists():
            resolved_cfg.unlink(missing_ok=True)

    return model, cfg, ckpt, data_dir, audio_dir, run_dir


def _make_dataset(
    data_dir: Path,
    audio_dir: Path,
    args: argparse.Namespace,
    split: str,
):
    return AudioWithParametersDataset(
        data_dir=str(data_dir),
        meta_file=args.meta_file,
        sample_rate=int(args.sample_rate),
        num_samples=None,
        split=split,
        seed=int(args.seed),
        split_strategy=args.split_strategy,
        parameter_key=args.parameter_key,
        expected_num_modes=args.expected_num_modes,
        audio_dir=str(audio_dir),
    )


def _unwrap(value):
    return value[0] if isinstance(value, tuple) else value


def _compute_latents(model, x, params):
    emb = None if model.encoder is None else model.encoder(x)
    modal = params
    if model.modal_autoencoder is not None:
        modal = _unwrap(
            model.modal_autoencoder(x, params)
            if model.modal_autoencoder_accepts_audio
            else model.modal_autoencoder(emb, params)
        )

    noise_lat = None
    if model.noise_autoencoder is not None:
        noise_lat = _unwrap(
            model.noise_autoencoder(x)
            if model.noise_autoencoder_accepts_audio
            else model.noise_autoencoder(emb)
        )

    transient_lat = None
    if model.transient_autoencoder is not None:
        transient_lat = _unwrap(
            model.transient_autoencoder(x)
            if model.transient_autoencoder_accepts_audio
            else model.transient_autoencoder(emb)
        )
    return modal, noise_lat, transient_lat


def _wav_bytes(waveform, sample_rate: int) -> bytes:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, waveform.detach().cpu(), int(sample_rate), format="wav")
    return buf.getvalue()


def _control_values() -> List[tuple[str, float]]:
    return [
        ("-1.00", -1.0),
        ("-0.50", -0.5),
        ("0.00", 0.0),
        ("+0.50", 0.5),
        ("+1.00", 1.0),
    ]


def _collect_transient_latents(
    model,
    dataset,
    device,
    count: int,
    seed: int,
) -> tuple[Any, Any]:
    idxs = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(int(seed)))
    idxs = idxs[: min(max(1, int(count)), len(dataset))].tolist()
    latents: List[Any] = []
    latent_shape = None

    with torch.no_grad():
        for idx in idxs:
            waveform, params, _ = dataset[idx]
            _, _, transient_lat = _compute_latents(
                model,
                waveform.unsqueeze(0).to(device),
                params.unsqueeze(0).to(device),
            )
            if transient_lat is None:
                continue
            tail = transient_lat.shape[1:]
            if latent_shape is None:
                latent_shape = tail
            elif tail != latent_shape:
                raise RuntimeError(f"inconsistent transient latent shape: {tail} vs {latent_shape}")
            latents.append(transient_lat.detach().reshape(1, -1).float().cpu())

    if not latents or latent_shape is None:
        raise RuntimeError("Could not collect transient latents from the selected model.")
    return torch.cat(latents, dim=0), latent_shape


def _axis_note(direction, centered_values) -> str:
    proj_std = float((centered_values @ direction).std(unbiased=False).item())
    axis_norm = float(direction.norm().item())
    return f"proj_std={proj_std:.4g}, direction_norm={axis_norm:.4g}"


def _build_pca_axes(values, count: int) -> List[LatentAxis]:
    centered = values - values.mean(dim=0, keepdim=True)
    dim = int(values.shape[1])
    if centered.shape[0] < 2:
        axes: List[LatentAxis] = []
        for i in range(min(int(count), dim)):
            direction = torch.zeros(dim, dtype=values.dtype)
            direction[i] = 1.0
            axes.append(
                LatentAxis(
                    name=f"pca-{i + 1}",
                    method="pca",
                    axis_flat=direction,
                    score=0.0,
                    note="fallback axis; fewer than two latent samples",
                )
            )
        return axes

    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.pow(2)
    denom = energy.sum().clamp_min(1e-8)
    total = min(int(count), vh.shape[0])
    axes = []
    for i in range(total):
        direction = vh[i]
        proj_std = (centered @ direction).std(unbiased=False).clamp_min(1e-8)
        ratio = float((energy[i] / denom).item())
        axes.append(
            LatentAxis(
                name=f"pca-{i + 1}",
                method="pca",
                axis_flat=direction * proj_std,
                score=ratio,
                note=f"explained={ratio:.2%}, {_axis_note(direction, centered)}",
            )
        )
    return axes


def _sample_filename(dataset, index: int) -> str:
    key = dataset.file_list[index]
    meta = dataset.metadata[key]
    return str(meta.get("orig_relpath") or meta.get("filename") or key)


def _build_one_sample_state(model, dataset, index: int, device) -> SampleState:
    waveform, params, length = dataset[int(index)]
    with torch.no_grad():
        modal, noise_lat, transient_lat = _compute_latents(
            model,
            waveform.unsqueeze(0).to(device),
            params.unsqueeze(0).to(device),
        )
    if transient_lat is None:
        raise RuntimeError("Selected model has no transient latent branch.")
    key = dataset.file_list[int(index)]
    return SampleState(
        sample_index=int(index),
        sample_key=str(key),
        sample_filename=_sample_filename(dataset, int(index)),
        length=int(length),
        target_waveform=waveform.detach().cpu(),
        modal_params=modal.detach(),
        noise_lat_base=None if noise_lat is None else noise_lat.detach(),
        transient_lat_base=transient_lat.detach(),
    )


def _build_sensitivity_axes(
    model,
    dataset,
    device,
    args: argparse.Namespace,
    latent_values,
    latent_shape,
    count: int,
) -> List[LatentAxis]:
    dim = int(latent_values.shape[1])
    if dim > int(args.max_sensitivity_dim):
        raise RuntimeError(
            f"sensitivity axis dim={dim} exceeds --max-sensitivity-dim={args.max_sensitivity_dim}. "
            "Increase the limit if you want to probe this full latent vector."
        )

    n_samples = min(max(1, int(args.sensitivity_samples)), len(dataset))
    probes = max(1, int(args.hutch_probes))
    idxs = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(int(args.seed) + 991),
    )[:n_samples].tolist()

    old_requires_grad = [p.requires_grad for p in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)

    gradients: List[Any] = []
    try:
        for idx in idxs:
            sample = _build_one_sample_state(model, dataset, int(idx), device)
            z0 = sample.transient_lat_base.detach().clone().requires_grad_(True)
            y = _render_from_transient_lat(model, sample, z0, detach=False, clamp=False)
            flat_y = y.reshape(-1)
            scale = float(flat_y.numel()) ** 0.5
            for probe_idx in range(probes):
                rand = torch.randint(
                    0,
                    2,
                    flat_y.shape,
                    device=flat_y.device,
                    dtype=torch.int64,
                )
                probe = rand.to(flat_y.dtype).mul(2).sub(1)
                scalar = (flat_y * probe).sum() / max(scale, 1.0)
                grad = torch.autograd.grad(
                    scalar,
                    z0,
                    retain_graph=probe_idx < probes - 1,
                    allow_unused=False,
                )[0]
                grad_flat = grad.detach().reshape(-1).float().cpu()
                norm = grad_flat.norm()
                if float(norm) <= 1e-10:
                    continue
                gradients.append(grad_flat / norm)
    finally:
        for param, requires_grad in zip(model.parameters(), old_requires_grad):
            param.requires_grad_(requires_grad)

    if not gradients:
        raise RuntimeError("No usable sensitivity gradients were collected.")

    # Low-rank SVD of collected output-gradient probes. This finds directions
    # that repeatedly produce strong waveform changes without building a dim^2
    # Jacobian metric, so larger latent vectors remain practical.
    grad_matrix = torch.stack(gradients, dim=0)
    _, singular_values, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
    centered = latent_values - latent_values.mean(dim=0, keepdim=True)

    axes: List[LatentAxis] = []
    total = min(max(1, int(count)), vh.shape[0])
    for out_idx in range(total):
        direction = vh[out_idx]
        direction = direction / direction.norm().clamp_min(1e-8)
        proj_std = (centered @ direction).std(unbiased=False).clamp_min(1e-8)
        score = float((singular_values[out_idx].pow(2) / max(1, len(gradients))).item())
        axes.append(
            LatentAxis(
                name=f"sensitivity-{out_idx + 1}",
                method="sensitivity",
                axis_flat=direction * proj_std,
                score=score,
                note=f"jacobian_energy={score:.4g}, probes={len(gradients)}, {_axis_note(direction, centered)}",
            )
        )
    return axes


def _group_latent_axes(
    axes: Sequence[LatentAxis],
    count: int,
    group_size: int,
) -> List[LatentAxis]:
    display_count = max(1, int(count))
    size = max(1, int(group_size))
    if size <= 1:
        return list(axes[:display_count])

    grouped: List[LatentAxis] = []
    for group_idx in range(display_count):
        group = list(axes[group_idx * size : (group_idx + 1) * size])
        if not group:
            break

        flats = [axis.axis_flat.float() for axis in group]
        dtype = flats[0].dtype
        weights = []
        for axis in group:
            score = max(0.0, float(axis.score))
            weights.append(score**0.5 if score > 0.0 else 1.0)
        weight = torch.tensor(weights, dtype=dtype).clamp_min(1e-8)
        stack = torch.stack(flats, dim=0)
        combined = (stack * weight[:, None]).sum(dim=0) / weight.pow(2).sum().sqrt().clamp_min(1e-8)

        method = group[0].method
        source_names = ", ".join(axis.name for axis in group)
        grouped.append(
            LatentAxis(
                name=f"{method}-group-{group_idx + 1}",
                method=method,
                axis_flat=combined,
                score=float(sum(float(axis.score) for axis in group)),
                note=f"combined {len(group)} raw {method} axes: {source_names}",
            )
        )
    return grouped


def _build_axes(
    model,
    axis_dataset,
    device,
    args: argparse.Namespace,
    latent_values,
    latent_shape,
) -> List[LatentAxis]:
    methods = ["pca", "sensitivity"] if args.methods == "both" else [args.methods]
    display_count = max(1, int(args.num_knobs))
    group_size = max(1, int(args.axis_group_size))
    raw_count = display_count * group_size
    axes: List[LatentAxis] = []
    if "pca" in methods:
        raw_axes = _build_pca_axes(latent_values, count=raw_count)
        axes.extend(_group_latent_axes(raw_axes, count=display_count, group_size=group_size))
    if "sensitivity" in methods:
        raw_axes = _build_sensitivity_axes(
            model=model,
            dataset=axis_dataset,
            device=device,
            args=args,
            latent_values=latent_values,
            latent_shape=latent_shape,
            count=raw_count,
        )
        axes.extend(_group_latent_axes(raw_axes, count=display_count, group_size=group_size))
    return axes


def _select_sample_indices(args: argparse.Namespace, dataset) -> List[int]:
    if args.sample_index:
        return [int(i) for i in args.sample_index]
    if args.sample_key:
        return [dataset.file_list.index(args.sample_key)]
    if args.sample_substr:
        query = args.sample_substr.lower()
        for i, key in enumerate(dataset.file_list):
            meta = dataset.metadata[key]
            haystack = " ".join(
                str(meta.get(name, ""))
                for name in ("filename", "orig_relpath", "sample_pack", "instrument")
            ).lower()
            if query in haystack or query in str(key).lower():
                return [i]
        raise KeyError(args.sample_substr)

    count = min(max(1, int(args.sample_count)), len(dataset))
    return random.Random(int(args.seed)).sample(range(len(dataset)), count)


def _build_sample_states(model, dataset, indices: Sequence[int], device) -> List[SampleState]:
    return [_build_one_sample_state(model, dataset, int(index), device) for index in indices]


def _render_from_transient_lat(model, sample: SampleState, transient_lat, detach: bool, clamp: bool):
    length = int(sample.length)
    y = model.modal_synth(sample.modal_params, length)

    noise = None
    if model.noise_synth is not None and sample.noise_lat_base is not None:
        try:
            noise = model.noise_synth(sample.noise_lat_base, length).unsqueeze(1)
        except Exception:
            noise = model.noise_synth(sample.noise_lat_base.transpose(1, 2), length).unsqueeze(1)
        if model.transient_takes_noise:
            y = y + noise

    if model.transient_synth is None:
        raise RuntimeError("Selected model has no transient_synth.")
    transient = model.transient_synth(y, transient_lat)
    y = y + transient if model.transient_parallel else transient

    if model.noise_synth is not None and noise is not None and not model.transient_takes_noise:
        y = y + noise

    out = y.squeeze(0)
    if clamp:
        out = out.clamp(-1.0, 1.0)
    if detach:
        out = out.detach().cpu()
    return out


def _render_neutral(model, sample: SampleState):
    return _render_from_transient_lat(
        model=model,
        sample=sample,
        transient_lat=sample.transient_lat_base,
        detach=True,
        clamp=True,
    )


def _render_with_axis(model, sample: SampleState, axis: LatentAxis, signed_value: float):
    return _render_with_axes(
        model=model,
        sample=sample,
        axis_values=[(axis, signed_value, 1.0)],
    )


def _render_with_axes(
    model,
    sample: SampleState,
    axis_values: Sequence[tuple[LatentAxis, float] | tuple[LatentAxis, float, float]],
):
    tlat = sample.transient_lat_base
    transient_lat = tlat.clone()
    for item in axis_values:
        if len(item) == 2:
            axis, signed_value = item
            strength = 1.0
        else:
            axis, signed_value, strength = item
        if abs(float(signed_value)) <= 1e-12:
            continue
        axis_delta = axis.axis_flat.to(device=tlat.device, dtype=tlat.dtype).reshape(tlat.shape[1:])
        transient_lat = transient_lat + float(signed_value) * float(strength) * axis_delta.unsqueeze(0)
    return _render_from_transient_lat(
        model=model,
        sample=sample,
        transient_lat=transient_lat,
        detach=True,
        clamp=True,
    )


def _render_index(
    args: argparse.Namespace,
    cfg: Path,
    ckpt: Path,
    data_dir: Path,
    audio_dir: Path,
    latent_shape,
    states: Sequence[SampleState],
    axes: Sequence[LatentAxis],
    values: Sequence[tuple[str, float]],
    used_seed: int,
    control_strength: float,
) -> bytes:
    rows: List[str] = []
    value_payload = json.dumps(
        [{"label": label, "value": float(signed)} for label, signed in values],
        ensure_ascii=False,
    )
    sensitivity_axes = [
        (idx, axis)
        for idx, axis in enumerate(axes)
        if axis.method == "sensitivity"
    ]
    pair_axes = sensitivity_axes[:2] if len(sensitivity_axes) >= 2 else list(enumerate(axes[:2]))

    for sid, sample in enumerate(states):
        target = (
            "<div class='cell target'><div class='lbl'>target</div>"
            f"<audio controls preload='none' src='/audio?kind=target&sid={sid}'></audio></div>"
        )
        baseline = (
            "<div class='cell base'><div class='lbl'>recon neutral</div>"
            f"<audio controls preload='none' src='/audio?kind=recon&sid={sid}&kid=0&value=0'></audio></div>"
        )
        axis_rows = []
        neutral_index = min(
            range(len(values)),
            key=lambda idx: abs(float(values[idx][1])),
        )
        neutral_label, neutral_value = values[neutral_index]

        if len(pair_axes) == 2:
            kid_1, axis_1 = pair_axes[0]
            kid_2, axis_2 = pair_axes[1]
            grid_buttons = []
            for row_idx, (label_2, value_2) in enumerate(reversed(values)):
                original_row_idx = len(values) - 1 - row_idx
                for col_idx, (label_1, value_1) in enumerate(values):
                    is_neutral = col_idx == neutral_index and original_row_idx == neutral_index
                    grid_buttons.append(
                        "<button type='button' "
                        f"data-i='{col_idx}' data-j='{original_row_idx}' "
                        f"data-value1='{value_1:.8f}' data-value2='{value_2:.8f}' "
                        f"class='{'is-active' if is_neutral else ''}' "
                        f"aria-label='{html.escape(axis_1.name)} {html.escape(label_1)}, "
                        f"{html.escape(axis_2.name)} {html.escape(label_2)}'>"
                        "</button>"
                    )
            axis_rows.append(
                "<section class='axis-row pair-row'>"
                "<div class='axis-meta'>"
                f"<strong>{html.escape(axis_1.name)} x {html.escape(axis_2.name)}</strong>"
                "<span>sensitivity pair</span>"
                f"<p>Axis X: {html.escape(axis_1.note)}</p>"
                f"<p>Axis Y: {html.escape(axis_2.note)}</p>"
                "</div>"
                "<div class='cell pair-cell' "
                f"data-sid='{sid}' data-kid1='{kid_1}' data-kid2='{kid_2}' "
                f"data-i='{neutral_index}' data-j='{neutral_index}'>"
                "<div class='knob-head'>"
                "<span class='audio-label'>2-axis transient latent control</span>"
                f"<strong class='pair-value'>x={html.escape(neutral_label)}, y={html.escape(neutral_label)}</strong>"
                "</div>"
                "<div class='pair-layout'>"
                f"<div class='axis-label y-label'>{html.escape(axis_2.name)}</div>"
                "<div class='pair-grid'>"
                f"{''.join(grid_buttons)}"
                "</div>"
                f"<div class='axis-label x-label'>{html.escape(axis_1.name)}</div>"
                "</div>"
                "<div class='pair-ticks'>"
                f"<span>Y {html.escape(values[0][0])}</span><span>Y {html.escape(values[-1][0])}</span>"
                f"<span>X {html.escape(values[0][0])}</span><span>X {html.escape(values[-1][0])}</span>"
                "</div>"
                f"<audio controls preload='none' src='/audio?kind=pair&sid={sid}&kid1={kid_1}&kid2={kid_2}&value1={neutral_value:.8f}&value2={neutral_value:.8f}'></audio>"
                "</div>"
                "</section>"
            )
        else:
            for kid, axis in enumerate(axes):
                axis_rows.append(
                    "<section class='axis-row'>"
                    "<div class='axis-meta'>"
                    f"<strong>{html.escape(axis.name)}</strong>"
                    f"<span>{html.escape(axis.method)}</span>"
                    f"<p>{html.escape(axis.note)}</p>"
                    "</div>"
                    "<div class='cell knob-cell' "
                    f"data-sid='{sid}' data-kid='{kid}' data-index='{neutral_index}'>"
                    "<div class='knob-head'>"
                    "<span class='audio-label'>Transient latent knob</span>"
                    f"<strong class='knob-value'>{html.escape(neutral_label)}</strong>"
                    "</div>"
                    f"<input class='knob' type='range' min='0' max='{len(values) - 1}' "
                    f"step='1' value='{neutral_index}' aria-label='Control value for {html.escape(axis.name)}'>"
                    "<div class='knob-ticks'>"
                    + "".join(
                        f"<button type='button' data-index='{idx}' class='{'is-active' if idx == neutral_index else ''}'>"
                        f"{html.escape(label)}</button>"
                        for idx, (label, _) in enumerate(values)
                    )
                    + "</div>"
                    f"<audio controls preload='none' src='/audio?kind=recon&sid={sid}&kid={kid}&value={neutral_value:.8f}'></audio>"
                    "</div>"
                    "</section>"
                )

        rows.append(
            "<article class='sample'>"
            "<header>"
            f"<div><strong>sample #{sid + 1}</strong>"
            f"<p>{html.escape(Path(sample.sample_filename).name)}</p></div>"
            f"<div class='quick'>{target}{baseline}</div>"
            "</header>"
            f"{''.join(axis_rows)}"
            "</article>"
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Transient Latent Controls</title>
  <style>
    :root {{ --ink:#171716; --muted:#686b63; --line:#d7ccb8; --paper:#f3eddf; --surface:#fffdf8; --accent:#a73f2d; --green:#1f7567; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; padding:22px; font-family:Segoe UI, sans-serif; color:var(--ink); background:radial-gradient(circle at top left,#fff8dc,transparent 28rem),var(--paper); }}
    h1 {{ margin:0 0 8px; font-family:Georgia, serif; font-size:clamp(30px,4vw,58px); line-height:.96; }}
    .mono, code {{ font-family:Cascadia Code, Consolas, monospace; }}
    .top {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:20px; align-items:end; margin-bottom:18px; }}
    .meta {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:8px; margin:16px 0; }}
    .pill {{ border:1px solid var(--line); background:var(--surface); padding:10px; min-width:0; }}
    .pill span {{ display:block; color:var(--muted); font-size:11px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }}
    .pill strong {{ display:block; margin-top:5px; overflow-wrap:anywhere; }}
    .sample {{ border:1px solid var(--line); background:var(--surface); margin:14px 0; box-shadow:0 10px 24px rgba(20,20,20,.08); }}
    .sample header {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:12px; align-items:start; padding:14px; border-bottom:1px solid var(--line); }}
    .sample header p {{ margin:6px 0 0; color:var(--muted); overflow-wrap:anywhere; }}
    .quick {{ display:grid; grid-template-columns:repeat(2,220px); gap:8px; }}
    .axis-row {{ display:grid; grid-template-columns:260px minmax(260px,1fr); gap:8px; padding:10px 14px; border-bottom:1px solid var(--line); }}
    .axis-row:last-child {{ border-bottom:0; }}
    .axis-meta strong {{ display:block; font-family:Cascadia Code, Consolas, monospace; }}
    .axis-meta span {{ display:inline-block; margin:6px 0; padding:3px 7px; color:#fff; background:var(--green); font-size:11px; font-weight:800; text-transform:uppercase; }}
    .axis-meta p {{ margin:0; color:var(--muted); font-size:12px; line-height:1.35; }}
    .cell {{ min-width:0; background:#f0e8dc; border:1px solid #d3c7b7; padding:8px; }}
    .target {{ background:#e8edf3; }}
    .base {{ background:#e7f0ed; }}
    .lbl {{ margin-bottom:5px; color:var(--muted); font-size:11px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }}
    .knob-cell {{ display:grid; gap:10px; background:#fff8ee; }}
    .knob-head {{ display:flex; justify-content:space-between; gap:10px; align-items:baseline; }}
    .knob-head strong {{ font-family:Cascadia Code, Consolas, monospace; color:var(--accent); }}
    .knob {{ width:100%; accent-color:var(--accent); }}
    .knob-ticks {{ display:grid; grid-template-columns:repeat({len(values)}, minmax(0,1fr)); gap:6px; }}
    .knob-ticks button {{ padding:7px 6px; border:1px solid var(--line); background:#fffdf8; color:var(--ink); font-weight:800; cursor:pointer; }}
    .knob-ticks button.is-active {{ border-color:var(--accent); background:var(--accent); color:#fff; }}
    .pair-cell {{ display:grid; gap:10px; background:#fff8ee; }}
    .pair-layout {{ display:grid; grid-template-columns:auto minmax(220px,360px); gap:8px 10px; align-items:center; }}
    .pair-grid {{ display:grid; grid-template-columns:repeat({len(values)}, 1fr); gap:5px; }}
    .pair-grid button {{ aspect-ratio:1; border:1px solid var(--line); background:#fffdf8; cursor:pointer; }}
    .pair-grid button:hover {{ border-color:var(--accent); }}
    .pair-grid button.is-active {{ background:var(--accent); border-color:var(--accent); box-shadow:0 0 0 3px rgba(167,63,45,.18); }}
    .axis-label {{ color:var(--muted); font-size:11px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }}
    .y-label {{ writing-mode:vertical-rl; transform:rotate(180deg); justify-self:end; }}
    .x-label {{ grid-column:2; justify-self:center; }}
    .pair-ticks {{ display:grid; grid-template-columns:repeat(4,1fr); gap:6px; color:var(--muted); font-size:11px; }}
    audio {{ width:100%; height:34px; }}
    a {{ color:var(--green); font-weight:800; }}
    @media (max-width:800px) {{ body {{ padding:10px; }} .top, .sample header, .axis-row {{ grid-template-columns:1fr; }} .quick {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="top">
    <div>
      <h1>Transient Latent Controls</h1>
      <p class="mono">value semantics: signed latent displacement. 0 is neutral; +/-1 moves along a displayed axis, scaled by control strength {control_strength:.2f}. Each displayed axis bundles {int(args.axis_group_size)} raw latent directions.</p>
    </div>
    <div><a href="/refresh">refresh samples</a></div>
  </div>
  <section class="meta">
    <div class="pill"><span>model</span><strong>{html.escape(ckpt.name)}</strong></div>
    <div class="pill"><span>latent shape</span><strong>{html.escape(str(tuple(latent_shape)))}</strong></div>
    <div class="pill"><span>axes</span><strong>{len(axes)} transient latent axes</strong></div>
    <div class="pill"><span>control strength</span><strong>{control_strength:.2f}x</strong></div>
  </section>
  {''.join(rows)}
  <script>
    const CONTROL_VALUES = {value_payload};

    function audioSrc(sid, kid, value) {{
      return `/audio?kind=recon&sid=${{sid}}&kid=${{kid}}&value=${{Number(value).toFixed(8)}}`;
    }}

    function pairAudioSrc(sid, kid1, kid2, value1, value2) {{
      return `/audio?kind=pair&sid=${{sid}}&kid1=${{kid1}}&kid2=${{kid2}}&value1=${{Number(value1).toFixed(8)}}&value2=${{Number(value2).toFixed(8)}}`;
    }}

    function setKnob(cell, nextIndex) {{
      const index = Math.max(0, Math.min(CONTROL_VALUES.length - 1, Number(nextIndex)));
      const item = CONTROL_VALUES[index];
      cell.dataset.index = String(index);
      cell.querySelector(".knob").value = String(index);
      cell.querySelector(".knob-value").textContent = item.label;
      cell.querySelectorAll(".knob-ticks button").forEach((button, buttonIndex) => {{
        button.classList.toggle("is-active", buttonIndex === index);
      }});
      const audio = cell.querySelector("audio");
      audio.src = audioSrc(cell.dataset.sid, cell.dataset.kid, item.value);
      audio.load();
      audio.play().catch(() => {{}});
    }}

    function setPair(cell, button) {{
      const value1 = Number(button.dataset.value1);
      const value2 = Number(button.dataset.value2);
      cell.dataset.i = button.dataset.i;
      cell.dataset.j = button.dataset.j;
      cell.querySelector(".pair-value").textContent =
        `x=${{CONTROL_VALUES[Number(button.dataset.i)].label}}, y=${{CONTROL_VALUES[Number(button.dataset.j)].label}}`;
      cell.querySelectorAll(".pair-grid button").forEach((item) => {{
        item.classList.toggle("is-active", item === button);
      }});
      const audio = cell.querySelector("audio");
      audio.src = pairAudioSrc(cell.dataset.sid, cell.dataset.kid1, cell.dataset.kid2, value1, value2);
      audio.load();
      audio.play().catch(() => {{}});
    }}

    document.querySelectorAll("audio").forEach((audio) => {{
      audio.addEventListener("play", () => {{
        document.querySelectorAll("audio").forEach((other) => {{
          if (other !== audio) other.pause();
        }});
      }});
    }});
    document.querySelectorAll(".knob-cell").forEach((cell) => {{
      cell.querySelector(".knob").addEventListener("input", (event) => {{
        setKnob(cell, event.target.value);
      }});
      cell.querySelectorAll(".knob-ticks button").forEach((button) => {{
        button.addEventListener("click", () => setKnob(cell, button.dataset.index));
      }});
    }});
    document.querySelectorAll(".pair-cell").forEach((cell) => {{
      cell.querySelectorAll(".pair-grid button").forEach((button) => {{
        button.addEventListener("click", () => setPair(cell, button));
      }});
    }});
  </script>
</body>
</html>"""
    return doc.encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=DEFAULT_RUN_DIR)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--audio-dir", type=str, default=None)
    parser.add_argument("--meta-file", type=str, default="metadata.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--axis-split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--split-strategy", type=str, default="sample_pack", choices=["sample_pack", "random"])
    parser.add_argument("--parameter-key", type=str, default="feature_file")
    parser.add_argument("--expected-num-modes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--methods", type=str, default="sensitivity", choices=["both", "pca", "sensitivity"])
    parser.add_argument("--axis-samples", type=int, default=512)
    parser.add_argument("--sensitivity-samples", type=int, default=64)
    parser.add_argument("--hutch-probes", type=int, default=4)
    parser.add_argument("--max-sensitivity-dim", type=int, default=65536)
    parser.add_argument("--num-knobs", type=int, default=2)
    parser.add_argument("--axis-group-size", type=int, default=4)
    parser.add_argument("--control-strength", type=float, default=16.0)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--sample-index", action="append", type=int, default=[])
    parser.add_argument("--sample-key", type=str, default=None)
    parser.add_argument("--sample-substr", type=str, default=None)
    parser.add_argument("--loss-cfg", type=str, default=None)
    parser.add_argument("--noise-encoder-backbone", type=str, default=None)
    parser.add_argument("--transient-encoder-backbone", type=str, default=None)
    parser.add_argument("--noise-encoder-cfg", type=str, default=None)
    parser.add_argument("--transient-encoder-cfg", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_runtime()
    dev = _device(args.device)
    model, cfg, ckpt, data_dir, audio_dir, _ = _resolve_model_and_data(args)
    model.eval().to(dev)

    axis_dataset = _make_dataset(data_dir, audio_dir, args, split=args.axis_split)
    test_dataset = _make_dataset(data_dir, audio_dir, args, split=args.split)
    latent_values, latent_shape = _collect_transient_latents(
        model=model,
        dataset=axis_dataset,
        device=dev,
        count=int(args.axis_samples),
        seed=int(args.seed),
    )
    axes = _build_axes(
        model=model,
        axis_dataset=axis_dataset,
        device=dev,
        args=args,
        latent_values=latent_values,
        latent_shape=latent_shape,
    )
    if not axes:
        raise RuntimeError("No transient latent axes were built.")
    values = _control_values()

    states: List[SampleState] = []
    used_seed = int(args.seed)
    cache: Dict[tuple[Any, ...], bytes] = {}
    target_cache: Dict[int, bytes] = {}
    lock = threading.Lock()

    def rebuild(refresh: bool = False) -> None:
        nonlocal cache, states, target_cache, used_seed
        used_seed = int(time.time_ns() % (2**31 - 1)) if refresh else int(args.seed)
        original_seed = args.seed
        args.seed = used_seed
        try:
            indices = _select_sample_indices(args, test_dataset)
        finally:
            args.seed = original_seed
        states = _build_sample_states(model, test_dataset, indices, dev)
        cache = {}
        target_cache = {}

    rebuild(False)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                page = _render_index(
                    args=args,
                    cfg=cfg,
                    ckpt=ckpt,
                    data_dir=data_dir,
                    audio_dir=audio_dir,
                    latent_shape=latent_shape,
                    states=states,
                    axes=axes,
                    values=values,
                    used_seed=used_seed,
                    control_strength=float(args.control_strength),
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return

            if parsed.path == "/refresh":
                with lock:
                    rebuild(True)
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()
                return

            if parsed.path != "/audio":
                self.send_error(404, "not found")
                return

            query = parse_qs(parsed.query)
            kind = query.get("kind", ["recon"])[0]
            sid = int(query.get("sid", ["0"])[0])
            kid = int(query.get("kid", ["0"])[0])
            signed_value = float(query.get("value", ["0"])[0])

            with lock:
                if sid < 0 or sid >= len(states):
                    self.send_error(404, "bad sid")
                    return
                if kind == "target":
                    if sid not in target_cache:
                        target_cache[sid] = _wav_bytes(states[sid].target_waveform, args.sample_rate)
                    audio = target_cache[sid]
                elif kind == "pair":
                    kid_1 = int(query.get("kid1", ["0"])[0])
                    kid_2 = int(query.get("kid2", ["1"])[0])
                    value_1 = float(query.get("value1", ["0"])[0])
                    value_2 = float(query.get("value2", ["0"])[0])
                    if kid_1 < 0 or kid_1 >= len(axes) or kid_2 < 0 or kid_2 >= len(axes):
                        self.send_error(404, "bad axis")
                        return
                    key = (
                        "pair",
                        sid,
                        kid_1,
                        kid_2,
                        round(float(value_1), 8),
                        round(float(value_2), 8),
                    )
                    if key not in cache:
                        if abs(value_1) <= 1e-12 and abs(value_2) <= 1e-12:
                            rendered = _render_neutral(model=model, sample=states[sid])
                        else:
                            rendered = _render_with_axes(
                                model=model,
                                sample=states[sid],
                                axis_values=[
                                    (axes[kid_1], value_1, float(args.control_strength)),
                                    (axes[kid_2], value_2, float(args.control_strength)),
                                ],
                            )
                        cache[key] = _wav_bytes(rendered, args.sample_rate)
                        if len(cache) > 512:
                            cache.pop(next(iter(cache.keys())), None)
                    audio = cache[key]
                else:
                    if kid < 0 or kid >= len(axes):
                        self.send_error(404, "bad axis")
                        return
                    key = (sid, kid, round(float(signed_value), 8))
                    if key not in cache:
                        if abs(float(signed_value)) <= 1e-12:
                            rendered = _render_neutral(model=model, sample=states[sid])
                        else:
                            rendered = _render_with_axis(
                                model=model,
                                sample=states[sid],
                                axis=axes[kid],
                                signed_value=float(signed_value) * float(args.control_strength),
                            )
                        cache[key] = _wav_bytes(rendered, args.sample_rate)
                        if len(cache) > 512:
                            cache.pop(next(iter(cache.keys())), None)
                    audio = cache[key]

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(audio)))
            self.end_headers()
            self.wfile.write(audio)

    print(f"[INFO] cfg={cfg}")
    print(f"[INFO] ckpt={ckpt}")
    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] audio_dir={audio_dir}")
    print(f"[INFO] device={dev}")
    print(f"[INFO] latent_shape={tuple(latent_shape)} axis_samples={len(latent_values)}")
    print(f"[INFO] axes={len(axes)} methods={args.methods} axis_group_size={args.axis_group_size}")
    print(f"[OK] transient latent control server: http://{args.host}:{args.port}/")
    ThreadingHTTPServer((args.host, int(args.port)), Handler).serve_forever()


if __name__ == "__main__":
    main()
