#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import random
import threading
import time
import types
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import torch
import torchaudio
import yaml

from drumblender.data.audio import AudioWithParametersDataset
from drumblender.utils.model import load_model


DEFAULT_SEED = 20260218
DEFAULT_SR = 48000
DEFAULT_SIGMA_MIN = 0.0
DEFAULT_SIGMA_MAX = 1.0
DEFAULT_SIGMA_STEPS = 5
DEFAULT_RANDOM_COUNT = 3
DEFAULT_DATA_DIR = "/workspace/datasets/modal_features/processed_modal_flat"


@dataclass
class AxisStats:
    pc1: torch.Tensor
    proj_std: float


@dataclass
class BasisStats:
    pcs: torch.Tensor   # [K, D]
    stds: torch.Tensor  # [K]
    mean: torch.Tensor  # [D]


@dataclass
class SampleState:
    sample_index: int
    sample_key: str
    sample_filename: str
    length: int
    target_waveform: torch.Tensor
    modal_params: torch.Tensor
    noise_lat_base: Optional[torch.Tensor]
    transient_lat_base: Optional[torch.Tensor]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _latest(paths: List[Path]) -> Optional[Path]:
    files = [p for p in paths if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _device(v: str) -> torch.device:
    return torch.device("cuda" if v == "auto" and torch.cuda.is_available() else ("cpu" if v == "auto" else v))


def _localize_path(path_str: Optional[str], root: Path) -> Optional[str]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_file():
        return str(p.resolve())
    s = str(path_str).replace("\\", "/")
    if "/workspace/drumblender/" in s:
        rel = s.split("/workspace/drumblender/", 1)[1]
        q = (root / rel).resolve()
        if q.is_file():
            return str(q)
    if not p.is_absolute():
        q = (root / p).resolve()
        if q.is_file():
            return str(q)
    return None


def _read_context(run_dir: Path) -> Dict:
    cands = list(run_dir.glob("run-context-*.json"))
    cands += list((run_dir / "training_context").glob("run-context-*.json"))
    cands += list((run_dir / "training_context" / "run_context").glob("run-context-*.json"))
    p = _latest(cands)
    if p is None:
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _infer_config(run_dir: Path, explicit: Optional[str], ctx: Dict, root: Path) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--config not found: {p}")
        return p
    c = _localize_path(ctx.get("cfg"), root)
    if c:
        return Path(c)
    fallback = root / "cfg" / "05_all_parallel.yaml"
    if fallback.is_file():
        return fallback.resolve()
    p = _latest(list((run_dir / "configs").glob("*.yaml")) + list((run_dir / "configs").glob("*.yml")))
    if p is None:
        raise FileNotFoundError("Could not infer config.")
    return p.resolve()


def _infer_ckpt(run_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.is_file():
            return p
    p = _latest(list(run_dir.rglob("*.ckpt")))
    if p is None:
        raise FileNotFoundError("Could not infer checkpoint in run_dir.")
    return p.resolve()


def _resolve_encoder_cfg(cfg_dir: Path, kind: str, backbone: str, explicit: Optional[str], root: Path) -> Optional[str]:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"{kind} encoder cfg not found: {p}")
        return str(p)
    if backbone == "soundstream":
        return None
    p = (cfg_dir / "upgrades" / "encoders" / f"{kind}_{backbone}_style.yaml").resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Missing encoder cfg: {p}")
    return str(p)


def _build_resolved_config(base_cfg: Path, loss_cfg: Optional[str], noise_backbone: str, transient_backbone: str, noise_cfg: Optional[str], transient_cfg: Optional[str], root: Path) -> Tuple[Path, bool]:
    cfg_dir = base_cfg.parent
    if not any([loss_cfg, noise_cfg, transient_cfg]) and noise_backbone == "soundstream" and transient_backbone == "soundstream":
        return base_cfg, False
    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    init_args = cfg.setdefault("model", {}).setdefault("init_args", {})
    if loss_cfg:
        init_args["loss_fn"] = str(Path(loss_cfg).resolve())
    ncfg = _resolve_encoder_cfg(cfg_dir, "noise", noise_backbone, noise_cfg, root)
    if ncfg:
        init_args["noise_autoencoder"] = ncfg
        init_args["noise_autoencoder_accepts_audio"] = True
    tcfg = _resolve_encoder_cfg(cfg_dir, "transient", transient_backbone, transient_cfg, root)
    if tcfg:
        init_args["transient_autoencoder"] = tcfg
        init_args["transient_autoencoder_accepts_audio"] = True
    tmp = cfg_dir / f".control_live_resolved_{int(time.time())}.yaml"
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return tmp, True


def _unwrap(x):
    return x[0] if isinstance(x, tuple) else x


def _compute_latents(model, x: torch.Tensor, p: torch.Tensor):
    emb = None if model.encoder is None else model.encoder(x)
    modal = p
    if model.modal_autoencoder is not None:
        modal = _unwrap(model.modal_autoencoder(x, p) if model.modal_autoencoder_accepts_audio else model.modal_autoencoder(emb, p))
    noise_lat = None
    if model.noise_autoencoder is not None:
        noise_lat = _unwrap(model.noise_autoencoder(x) if model.noise_autoencoder_accepts_audio else model.noise_autoencoder(emb))
    transient_lat = None
    if model.transient_autoencoder is not None:
        transient_lat = _unwrap(model.transient_autoencoder(x) if model.transient_autoencoder_accepts_audio else model.transient_autoencoder(emb))
    return modal, noise_lat, transient_lat


def _first_pc(X: torch.Tensor) -> AxisStats:
    m = X.mean(dim=0, keepdim=True)
    Xc = X - m
    if Xc.shape[0] < 2:
        v = torch.zeros(X.shape[1], device=X.device, dtype=X.dtype)
        v[0] = 1.0
        return AxisStats(v, 1.0)
    _, _, vh = torch.linalg.svd(Xc, full_matrices=False)
    pc = vh[0]
    std = float((Xc @ pc).std(unbiased=False).item())
    return AxisStats(pc, 1.0 if std < 1e-8 else std)


def _topk_pcs(X: torch.Tensor, k: int) -> BasisStats:
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    if Xc.shape[0] < 2:
        d = X.shape[1]
        pc = torch.zeros(1, d, device=X.device, dtype=X.dtype)
        pc[0, 0] = 1.0
        return BasisStats(
            pcs=pc,
            stds=torch.ones(1, device=X.device, dtype=X.dtype),
            mean=mean.squeeze(0),
        )

    _, _, vh = torch.linalg.svd(Xc, full_matrices=False)
    kk = max(1, min(int(k), vh.shape[0]))
    pcs = vh[:kk]
    proj = Xc @ pcs.t()
    stds = proj.std(dim=0, unbiased=False).clamp_min(1e-8)
    return BasisStats(pcs=pcs, stds=stds, mean=mean.squeeze(0))


def _sym_decorrelation(W: torch.Tensor) -> torch.Tensor:
    M = W @ W.t()
    vals, vecs = torch.linalg.eigh(M)
    vals = torch.clamp(vals, min=1e-8)
    inv_sqrt = vecs @ torch.diag(torch.rsqrt(vals)) @ vecs.t()
    return inv_sqrt @ W


def _fastica_axes(
    Xc: torch.Tensor,
    n_components: int = 2,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> Optional[torch.Tensor]:
    """
    Return ICA axes in original embedding space as [n_components, D].
    Uses symmetric FastICA on PCA-whitened features.
    """
    n, d = Xc.shape
    m = min(int(n_components), d, max(1, n - 1))
    if n < 4 or m < 1:
        return None

    # PCA + whitening to m dimensions
    _, s, vh = torch.linalg.svd(Xc, full_matrices=False)
    s = torch.clamp(s[:m], min=1e-6)
    V = vh[:m]  # [m, D]
    scale = math.sqrt(max(n - 1, 1))
    Z = (Xc @ V.t()) / s.unsqueeze(0) * scale  # [N, m]

    g = torch.Generator(device=Xc.device)
    g.manual_seed(20260218)
    W = torch.randn(m, m, device=Xc.device, dtype=Xc.dtype, generator=g)
    W = _sym_decorrelation(W)

    for _ in range(max_iter):
        WX = Z @ W.t()                 # [N, m]
        GWX = torch.tanh(WX)           # nonlinearity
        Gp = 1.0 - GWX.pow(2)          # derivative

        W1 = (GWX.t() @ Z) / float(n) - torch.diag(Gp.mean(dim=0)) @ W
        W1 = _sym_decorrelation(W1)

        lim = torch.max(torch.abs(torch.abs(torch.diagonal(W1 @ W.t())) - 1.0))
        W = W1
        if float(lim) < tol:
            break

    # Map ICA directions back to original embedding space.
    B = (V.t() / s.unsqueeze(0)) * scale     # [D, m]
    axes = (B @ W.t()).t()                   # [m, D]
    axes = torch.nn.functional.normalize(axes, dim=1)
    return axes


def _build_pca_ica3_basis(X: torch.Tensor) -> BasisStats:
    """
    Build 3-axis basis:
    axis0 = PCA #1
    axis1 = ICA #1
    axis2 = ICA #2
    """
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    d = X.shape[1]

    # PCA #1
    _, _, vh = torch.linalg.svd(Xc, full_matrices=False)
    pc1 = vh[0:1]  # [1, D]

    # ICA #1,#2
    ica_axes = _fastica_axes(Xc, n_components=2)
    if ica_axes is None or ica_axes.shape[0] < 2:
        # Fallback: PCA #2,#3 (or padded)
        pca_rest = vh[1:3]
        if pca_rest.shape[0] < 2:
            pad = torch.zeros(2 - pca_rest.shape[0], d, device=X.device, dtype=X.dtype)
            if pca_rest.shape[0] == 0:
                pad[0, 0] = 1.0
            pca_rest = torch.cat([pca_rest, pad], dim=0)
        ica_axes = pca_rest
    else:
        ica_axes = ica_axes[:2]

    axes = torch.cat([pc1, ica_axes], dim=0)  # [3, D]

    # Make axes numerically stable and compute scale per axis.
    axes = torch.nn.functional.normalize(axes, dim=1)
    proj = Xc @ axes.t()
    stds = proj.std(dim=0, unbiased=False).clamp_min(1e-8)

    return BasisStats(pcs=axes, stds=stds, mean=mean.squeeze(0))


def _build_transient_axis(model, ds: AudioWithParametersDataset, dev: torch.device, n: int, seed: int) -> Optional[AxisStats]:
    idxs = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[: min(max(1, n), len(ds))].tolist()
    tlist = []
    with torch.no_grad():
        for i in idxs:
            w, p, _ = ds[i]
            _, _, t = _compute_latents(model, w.unsqueeze(0).to(dev), p.unsqueeze(0).to(dev))
            if t is None:
                continue
            if t.ndim == 1:
                t = t.unsqueeze(0)
            tlist.append(t.detach())
    if not tlist:
        return None
    return _first_pc(torch.cat(tlist, dim=0))


def _build_transient_basis(
    model,
    ds: AudioWithParametersDataset,
    dev: torch.device,
    n: int,
    seed: int,
    k: int,
    basis_type: str = "pca_ica3",
) -> Optional[BasisStats]:
    idxs = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[
        : min(max(1, n), len(ds))
    ].tolist()
    tlist = []
    with torch.no_grad():
        for i in idxs:
            w, p, _ = ds[i]
            _, _, t = _compute_latents(
                model, w.unsqueeze(0).to(dev), p.unsqueeze(0).to(dev)
            )
            if t is None or t.ndim != 2:
                continue
            tlist.append(t.detach())
    if not tlist:
        return None
    X = torch.cat(tlist, dim=0)
    if basis_type == "pca_ica3":
        return _build_pca_ica3_basis(X)
    return _topk_pcs(X, k=max(1, int(k)))


def _wav_bytes(w: torch.Tensor, sr: int) -> bytes:
    if w.ndim == 1:
        w = w.unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, w.detach().cpu(), sr, format="wav")
    return buf.getvalue()


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _tcn_controls_from_sigma(args: argparse.Namespace, sigma: float) -> Dict[str, float]:
    s = max(0.0, min(1.0, float(sigma)))
    amount = float(args.tcn_amount)
    length_ms = float(args.tcn_length_ms)
    tone = float(args.tcn_tone)

    if args.tcn_map == "amount":
        amount = _lerp(float(args.tcn_amount_min), float(args.tcn_amount_max), s)
    elif args.tcn_map == "length":
        length_ms = _lerp(float(args.tcn_length_min_ms), float(args.tcn_length_max_ms), s)
    elif args.tcn_map == "tone":
        tone = _lerp(float(args.tcn_tone_min), float(args.tcn_tone_max), s)
    elif args.tcn_map == "all":
        amount = _lerp(float(args.tcn_amount_min), float(args.tcn_amount_max), s)
        length_ms = _lerp(float(args.tcn_length_min_ms), float(args.tcn_length_max_ms), s ** 1.35)
        tone_s = 0.5 + 0.5 * math.tanh((s - 0.5) * 3.0)
        tone = _lerp(float(args.tcn_tone_min), float(args.tcn_tone_max), tone_s)

    return {"amount": amount, "length_ms": length_ms, "tone": tone}


def _build_decay_env(
    length: int,
    sample_rate: int,
    hold_ms: float,
    decay_ms: float,
    device,
    dtype,
) -> torch.Tensor:
    n = max(1, int(length))
    t = torch.arange(n, device=device, dtype=dtype)
    hold = max(0, int(float(hold_ms) * sample_rate / 1000.0))
    tau = max(1.0, float(decay_ms) * sample_rate / 1000.0)
    env = torch.ones(n, device=device, dtype=dtype)
    if hold < n:
        env[hold:] = torch.exp(-(t[hold:] - float(hold)) / tau)
    return env


def _apply_tone_tilt(
    x: torch.Tensor,
    tone: float,
    sample_rate: int,
    pivot_hz: float = 2500.0,
) -> torch.Tensor:
    if abs(float(tone)) < 1e-6:
        return x
    X = torch.fft.rfft(x, dim=-1)
    n_bins = X.shape[-1]
    freqs = torch.linspace(
        0.0, float(sample_rate) * 0.5, n_bins, device=x.device, dtype=x.dtype
    )
    ratio = torch.clamp(freqs / float(pivot_hz), min=1e-3)
    gain = ratio.pow(float(tone)).clamp(0.25, 4.0)
    X = X * gain.view(1, 1, -1)
    return torch.fft.irfft(X, n=x.shape[-1], dim=-1)


def _apply_tcn_controls(
    t: torch.Tensor,
    sample_rate: int,
    amount: float,
    length_ms: float,
    tone: float,
    hold_ms: float,
) -> torch.Tensor:
    y = _apply_tone_tilt(t, tone=float(tone), sample_rate=int(sample_rate))
    env = _build_decay_env(
        length=y.shape[-1],
        sample_rate=int(sample_rate),
        hold_ms=float(hold_ms),
        decay_ms=float(length_ms),
        device=y.device,
        dtype=y.dtype,
    )
    y = y * env.view(1, 1, -1)
    y = y * float(amount)
    return y


def _apply_film_magic(
    embedding: torch.Tensor,
    sigma: float,
    basis: BasisStats,
    strength: float,
    drive: float,
    bend: float,
    sine_mix: float,
    char_gate: float,
    delta_clip: float,
) -> torch.Tensor:
    if embedding.ndim != 2:
        return embedding

    z = max(-1.0, min(1.0, (float(sigma) - 0.5) * 2.0))

    pcs = basis.pcs.to(device=embedding.device, dtype=embedding.dtype)
    stds = basis.stds.to(device=embedding.device, dtype=embedding.dtype)
    mean = basis.mean.to(device=embedding.device, dtype=embedding.dtype)
    k = pcs.shape[0]

    if k < 1:
        return embedding

    # Sample-specific coefficients in PCA coordinates (character-aware).
    centered = embedding - mean.unsqueeze(0)  # [B, D]
    coeff = (centered @ pcs.t()) / stds.unsqueeze(0)  # [B, K]

    # Strong character dims react more (gate in [0, 1]).
    gate = torch.sigmoid(float(char_gate) * (coeff.abs() - 0.35))

    # Nonlinear warp terms (all are zero when z=0).
    term1 = float(z) * torch.tanh(float(drive) * coeff)
    term2 = float(bend) * (float(z) ** 3) * torch.sign(coeff) * coeff.abs().pow(1.15)
    term3 = float(sine_mix) * math.sin(math.pi * float(z)) * torch.sin(coeff)

    delta_norm = gate * (term1 + term2 + term3)
    delta_coeff = float(strength) * delta_norm * stds.unsqueeze(0)

    if float(delta_clip) > 0:
        clip = float(delta_clip) * stds.unsqueeze(0)
        delta_coeff = torch.clamp(delta_coeff, -clip, clip)

    # Reconstruct delta in embedding space using the same principal directions.
    delta_emb = delta_coeff @ pcs
    return embedding + delta_emb


def _soft_limit(x: torch.Tensor, limit: float) -> torch.Tensor:
    """
    Smooth limiter with gentler high-amplitude behavior than tanh/clamp.
    """
    lim = torch.as_tensor(float(limit), device=x.device, dtype=x.dtype).clamp_min(1e-6)
    return lim * x / (lim + x.abs() + 1e-8)


class _FiLMAffinePatcher:
    """
    Inference-only monkey patch of FiLM.forward in transient TCN blocks.
    Directly modulates gamma/beta using one knob (sigma).
    """

    def __init__(
        self,
        transient_synth,
        args: argparse.Namespace,
        sigma: float,
        basis: Optional[BasisStats] = None,
    ):
        self.transient_synth = transient_synth
        self.args = args
        z_raw = max(-1.0, min(1.0, (float(sigma) - 0.5) * 2.0))
        # Keep mid-range z responsive while still allowing edge emphasis.
        z_curve = max(0.0, float(args.film_affine_z_curve))
        curve_amt = z_curve / (1.0 + z_curve)
        z_soft = max(-1.0, min(1.0, z_raw + curve_amt * (z_raw**3)))
        z_pow = max(0.5, float(args.film_affine_z_power))
        # Blend linear and power mapping to avoid a dead zone around z=0.
        z_mag = 0.6 * abs(z_soft) + 0.4 * (abs(z_soft) ** z_pow)
        self.z = math.copysign(z_mag, z_soft)
        self.basis = basis
        self.patches: List[Tuple[object, object]] = []

    def __enter__(self):
        if self.transient_synth is None or not hasattr(self.transient_synth, "tcn"):
            return self
        net = getattr(self.transient_synth.tcn, "net", None)
        if net is None:
            return self

        n_layers = max(1, len(net))
        for li, layer in enumerate(net):
            film = getattr(layer, "film", None)
            if film is None:
                continue
            original_forward = film.forward
            depth = 0.35 + 0.65 * (float(li) / float(max(1, n_layers - 1)))
            z = self.z
            args = self.args

            def _patched_forward(
                this,
                x,
                film_embedding,
                _depth=depth,
                _z=z,
                _args=args,
                _basis=self.basis,
            ):
                film_affine = this.net(film_embedding)
                gamma, beta = film_affine.chunk(2, dim=-1)
                if this.use_batch_norm:
                    x = this.norm(x)

                B = film_embedding.shape[0]
                z1 = torch.full(
                    (B, 1),
                    float(_z),
                    device=film_embedding.device,
                    dtype=film_embedding.dtype,
                )

                if _basis is not None and _basis.pcs.numel() > 0:
                    pcs = _basis.pcs.to(
                        device=film_embedding.device, dtype=film_embedding.dtype
                    )
                    stds = _basis.stds.to(
                        device=film_embedding.device, dtype=film_embedding.dtype
                    )
                    mean = _basis.mean.to(
                        device=film_embedding.device, dtype=film_embedding.dtype
                    )
                    coeff = (film_embedding - mean.unsqueeze(0)) @ pcs.t()
                    coeff = coeff / stds.unsqueeze(0)

                    c1 = coeff[..., 0:1]
                    c2 = coeff[..., 1:2] if coeff.shape[-1] > 1 else torch.zeros_like(c1)
                    c3 = coeff[..., 2:3] if coeff.shape[-1] > 2 else torch.zeros_like(c1)
                    latent_scale = torch.sigmoid(
                        float(_args.film_affine_latent_gain) * c1
                    )
                    latent_shape = torch.tanh(0.6 * c2)
                    latent_edge = torch.tanh(0.6 * c3)
                else:
                    latent_scale = torch.ones_like(z1)
                    latent_shape = torch.zeros_like(z1)
                    latent_edge = torch.zeros_like(z1)

                layer_w = max(0.10, 1.0 - float(_args.film_affine_tail_protect) * _depth)
                latent_mul = torch.clamp(0.85 + 0.30 * (latent_scale - 0.5), 0.65, 1.15)
                z_eff = z1 * latent_mul * layer_w

                drive = float(_args.film_affine_drive)
                drive_z = float(_args.film_affine_drive_z)
                drive_eff = torch.clamp(drive + drive_z * z_eff.abs(), 0.05, 8.0)

                # Near-linear shaping: keeps mid-z audible, limits edge explosion.
                z_curve_eff = z_eff * (0.80 + 0.20 * z_eff.abs())
                z_sq_signed = z_eff.abs() * z_eff

                g_gain = 1.0 + _depth * (
                    float(_args.film_affine_gamma_gain) * z_eff
                    + float(_args.film_affine_gamma_curve)
                    * (z_curve_eff + 0.10 * latent_edge * z_sq_signed)
                )
                b_gain = 1.0 + _depth * (
                    float(_args.film_affine_beta_gain) * z_eff
                    + float(_args.film_affine_beta_curve)
                    * (z_curve_eff + 0.10 * latent_shape * z_eff)
                )
                b_shift_scale = torch.clamp(1.0 - 0.35 * z_eff.abs(), 0.40, 1.0)
                b_shift = _depth * float(_args.film_affine_beta_shift) * z_eff * b_shift_scale

                gamma_mod = 1.0 + (gamma - 1.0) * g_gain
                beta_mod = beta * b_gain + b_shift

                gamma_mod = 1.0 + _soft_limit(drive_eff * (gamma_mod - 1.0), limit=1.8)
                beta_mod = _soft_limit(drive_eff * beta_mod, limit=1.6)

                mix = torch.clamp(
                    float(_args.film_affine_strength) * torch.sqrt(z_eff.abs() + 1e-8),
                    0.0,
                    1.0,
                )
                gamma_new = gamma + mix * (gamma_mod - gamma)
                beta_new = beta + mix * (beta_mod - beta)

                clamp_v = float(_args.film_affine_clamp)
                if clamp_v > 0:
                    gamma_new = torch.clamp(gamma_new, -clamp_v, clamp_v)
                    beta_new = torch.clamp(beta_new, -clamp_v, clamp_v)

                return gamma_new[..., None] * x + beta_new[..., None]

            film.forward = types.MethodType(_patched_forward, film)
            self.patches.append((film, original_forward))
        return self

    def __exit__(self, exc_type, exc, tb):
        for film, original_forward in reversed(self.patches):
            film.forward = original_forward
        self.patches.clear()
        return False


def _render_with_module(
    *,
    model,
    sample: SampleState,
    sigma: float,
    args: argparse.Namespace,
    transient_axis: Optional[AxisStats],
    transient_basis: Optional[BasisStats],
) -> torch.Tensor:
    tlat = sample.transient_lat_base
    if args.module == "transient_pca" and tlat is not None and transient_axis is not None:
        centered = ((float(sigma) - 0.5) * 2.0) * float(args.strength_max)
        d = centered * float(args.transient_scale) * float(transient_axis.proj_std) * transient_axis.pc1
        tlat = tlat + d.view(1, -1)
    elif args.module == "tcn_film_controls" and tlat is not None and transient_basis is not None:
        tlat = _apply_film_magic(
            embedding=tlat,
            sigma=float(sigma),
            basis=transient_basis,
            strength=float(args.film_strength),
            drive=float(args.film_drive),
            bend=float(args.film_bend),
            sine_mix=float(args.film_sine_mix),
            char_gate=float(args.film_char_gate),
            delta_clip=float(args.film_delta_clip),
        )

    L = int(sample.length)
    y = model.modal_synth(sample.modal_params, L)

    noise = None
    if model.noise_synth is not None and sample.noise_lat_base is not None:
        try:
            noise = model.noise_synth(sample.noise_lat_base, L).unsqueeze(1)
        except Exception:
            noise = model.noise_synth(sample.noise_lat_base.transpose(1, 2), L).unsqueeze(1)
        if model.transient_takes_noise:
            y = y + noise

    if model.transient_synth is not None and tlat is not None:
        if args.module == "tcn_film_affine_controls":
            with _FiLMAffinePatcher(
                model.transient_synth, args, float(sigma), basis=transient_basis
            ):
                t = model.transient_synth(y, tlat)
        else:
            t = model.transient_synth(y, tlat)
        if args.module in ("tcn_controls", "tcn_internal_controls"):
            ctrl = _tcn_controls_from_sigma(args, float(sigma))
            t = _apply_tcn_controls(
                t=t,
                sample_rate=int(args.sample_rate),
                amount=float(ctrl["amount"]),
                length_ms=float(ctrl["length_ms"]),
                tone=float(ctrl["tone"]),
                hold_ms=float(args.tcn_hold_ms),
            )
        y = (y + t) if model.transient_parallel else t

    if model.noise_synth is not None and noise is not None and (not model.transient_takes_noise):
        y = y + noise

    return y.squeeze(0).detach().cpu().clamp(-1.0, 1.0)


def _sigma_label(args: argparse.Namespace, sigma: float) -> str:
    if args.module == "tcn_film_controls":
        z = (float(sigma) - 0.5) * 2.0
        return f"film-z={z:+.2f} ({args.film_basis})"
    if args.module == "tcn_film_affine_controls":
        z = (float(sigma) - 0.5) * 2.0
        return f"film-affine-z={z:+.2f} ({args.film_basis})"
    if args.module in ("tcn_controls", "tcn_internal_controls"):
        ctrl = _tcn_controls_from_sigma(args, float(sigma))
        if args.tcn_map == "amount":
            return f"amount={ctrl['amount']:.2f}"
        if args.tcn_map == "length":
            return f"length={ctrl['length_ms']:.1f}ms"
        if args.tcn_map == "tone":
            return f"tone={ctrl['tone']:.2f}"
        return f"a={ctrl['amount']:.2f}, l={ctrl['length_ms']:.0f}ms, t={ctrl['tone']:.2f}"
    return f"sigma={sigma:.2f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Live recon control (transient-latent only)")
    p.add_argument("run_dir", type=str)
    p.add_argument(
        "--module",
        type=str,
        default="transient_pca",
        choices=[
            "none",
            "transient_pca",
            "tcn_controls",
            "tcn_internal_controls",
            "tcn_film_controls",
            "tcn_film_affine_controls",
        ],
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--meta-file", type=str, default="metadata.json")
    p.add_argument("--split-strategy", type=str, default="sample_pack")
    p.add_argument("--parameter-key", type=str, default="feature_file")
    p.add_argument("--expected-num-modes", type=int, default=64)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    p.add_argument("--sigma-min", type=float, default=DEFAULT_SIGMA_MIN)
    p.add_argument("--sigma-max", type=float, default=DEFAULT_SIGMA_MAX)
    p.add_argument("--sigma-steps", type=int, default=DEFAULT_SIGMA_STEPS)
    p.add_argument("--random-count", type=int, default=DEFAULT_RANDOM_COUNT)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--axis-samples", type=int, default=256)
    p.add_argument("--strength-max", type=float, default=2.0)
    p.add_argument("--transient-scale", type=float, default=1.0)
    p.add_argument("--tcn-map", type=str, default="all", choices=["amount", "length", "tone", "all"])
    p.add_argument("--tcn-amount", type=float, default=1.0)
    p.add_argument("--tcn-amount-min", type=float, default=0.2)
    p.add_argument("--tcn-amount-max", type=float, default=3.0)
    p.add_argument("--tcn-length-ms", type=float, default=40.0)
    p.add_argument("--tcn-length-min-ms", type=float, default=2.0)
    p.add_argument("--tcn-length-max-ms", type=float, default=400.0)
    p.add_argument("--tcn-tone", type=float, default=0.0)
    p.add_argument("--tcn-tone-min", type=float, default=-3.0)
    p.add_argument("--tcn-tone-max", type=float, default=3.0)
    p.add_argument("--tcn-hold-ms", type=float, default=0.0)
    p.add_argument("--film-k", type=int, default=4)
    p.add_argument("--film-basis", type=str, default="pca_ica3", choices=["pca_ica3", "pca"])
    p.add_argument("--film-strength", type=float, default=3.0)
    p.add_argument("--film-drive", type=float, default=2.5)
    p.add_argument("--film-bend", type=float, default=1.2)
    p.add_argument("--film-sine-mix", type=float, default=0.8)
    p.add_argument("--film-char-gate", type=float, default=2.0)
    p.add_argument("--film-delta-clip", type=float, default=4.0)
    p.add_argument("--film-affine-drive", type=float, default=1.2)
    p.add_argument("--film-affine-drive-z", type=float, default=0.25)
    p.add_argument("--film-affine-gamma-gain", type=float, default=0.55)
    p.add_argument("--film-affine-gamma-curve", type=float, default=0.25)
    p.add_argument("--film-affine-beta-gain", type=float, default=0.70)
    p.add_argument("--film-affine-beta-curve", type=float, default=0.30)
    p.add_argument("--film-affine-beta-shift", type=float, default=0.08)
    p.add_argument("--film-affine-strength", type=float, default=0.35)
    p.add_argument("--film-affine-z-curve", type=float, default=1.2)
    p.add_argument("--film-affine-z-power", type=float, default=1.35)
    p.add_argument("--film-affine-tail-protect", type=float, default=0.65)
    p.add_argument("--film-affine-latent-gain", type=float, default=1.4)
    p.add_argument("--film-affine-clamp", type=float, default=3.0)
    p.add_argument("--loss-cfg", type=str, default=None)
    p.add_argument("--noise-encoder-backbone", type=str, default=None)
    p.add_argument("--transient-encoder-backbone", type=str, default=None)
    p.add_argument("--noise-encoder-cfg", type=str, default=None)
    p.add_argument("--transient-encoder-cfg", type=str, default=None)
    p.add_argument("--sample-index", type=int, default=None)
    p.add_argument("--sample-key", type=str, default=None)
    p.add_argument("--sample-substr", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    root = _repo_root()
    run_dir = Path(args.run_dir).expanduser().resolve()
    ctx = _read_context(run_dir)
    cfg = _infer_config(run_dir, args.config, ctx, root)
    ckpt = _infer_ckpt(run_dir, args.ckpt)
    data_dir = Path(args.data_dir or _localize_path(ctx.get("data_dir"), root) or DEFAULT_DATA_DIR).expanduser().resolve()
    loss_cfg = args.loss_cfg or _localize_path(ctx.get("loss_cfg") if str(ctx.get("loss_upgrade", "")).lower() == "on" else None, root)
    noise_backbone = args.noise_encoder_backbone or str(ctx.get("noise_encoder_backbone") or "soundstream")
    transient_backbone = args.transient_encoder_backbone or str(ctx.get("transient_encoder_backbone") or "soundstream")
    noise_cfg = args.noise_encoder_cfg or _localize_path(ctx.get("noise_encoder_cfg"), root)
    transient_cfg = args.transient_encoder_cfg or _localize_path(ctx.get("transient_encoder_cfg"), root)

    resolved_cfg, is_tmp = _build_resolved_config(cfg, loss_cfg, noise_backbone, transient_backbone, noise_cfg, transient_cfg, root)
    try:
        model, _ = load_model(str(resolved_cfg), str(ckpt), include_data=False)
    finally:
        if is_tmp and resolved_cfg.exists():
            resolved_cfg.unlink(missing_ok=True)

    dev = _device(args.device)
    model.eval().to(dev)

    axis_ds = AudioWithParametersDataset(
        data_dir=str(data_dir),
        meta_file=args.meta_file,
        sample_rate=args.sample_rate,
        num_samples=None,
        split="train",
        seed=args.seed,
        split_strategy=args.split_strategy,
        parameter_key=args.parameter_key,
        expected_num_modes=args.expected_num_modes,
    )
    test_ds = AudioWithParametersDataset(
        data_dir=str(data_dir),
        meta_file=args.meta_file,
        sample_rate=args.sample_rate,
        num_samples=None,
        split="test",
        seed=args.seed,
        split_strategy=args.split_strategy,
        parameter_key=args.parameter_key,
        expected_num_modes=args.expected_num_modes,
    )
    taxis = (
        _build_transient_axis(model, axis_ds, dev, args.axis_samples, args.seed)
        if args.module == "transient_pca"
        else None
    )
    tbasis = (
        _build_transient_basis(
            model=model,
            ds=axis_ds,
            dev=dev,
            n=args.axis_samples,
            seed=args.seed,
            k=args.film_k,
            basis_type=args.film_basis,
        )
        if args.module in ("tcn_film_controls", "tcn_film_affine_controls")
        else None
    )
    if args.module == "transient_pca" and taxis is None:
        raise RuntimeError("Transient axis could not be built.")
    if args.module in ("tcn_film_controls", "tcn_film_affine_controls") and tbasis is None:
        raise RuntimeError("Transient FiLM basis could not be built.")

    sigmas = [float(v) for v in torch.linspace(args.sigma_min, args.sigma_max, steps=max(2, args.sigma_steps)).tolist()]
    sigma_variants = [s for s in sigmas if abs(s - 0.5) > 1e-9] or [0.0, 1.0]
    cache: Dict[Tuple[int, float], bytes] = {}
    target_cache: Dict[int, bytes] = {}
    lock = threading.Lock()
    states: List[SampleState] = []
    used_seed = args.seed

    def rebuild(refresh: bool) -> None:
        nonlocal states, cache, target_cache, used_seed
        if args.sample_key is not None:
            idxs = [test_ds.file_list.index(args.sample_key)]
        elif args.sample_substr is not None:
            q = args.sample_substr.lower(); idxs = []
            for i, key in enumerate(test_ds.file_list):
                m = test_ds.metadata[key]
                if q in str(m.get("filename", "")).lower() or q in str(m.get("orig_relpath", "")).lower() or q in key.lower():
                    idxs = [i]; break
            if not idxs: raise KeyError(args.sample_substr)
        elif args.sample_index is not None:
            idxs = [int(args.sample_index)]
        else:
            k = min(max(1, args.random_count), len(test_ds))
            used_seed = int(time.time_ns() % (2**31 - 1)) if refresh else int(args.seed)
            idxs = random.Random(used_seed).sample(range(len(test_ds)), k)
        out = []
        with torch.no_grad():
            for i in idxs:
                w, prm, L = test_ds[i]
                modal, nlat, tlat = _compute_latents(model, w.unsqueeze(0).to(dev), prm.unsqueeze(0).to(dev))
                key = test_ds.file_list[i]; meta = test_ds.metadata[key]
                out.append(SampleState(i, str(key), str(meta.get("filename", "")), int(L), w.detach().cpu(), modal.detach(), None if nlat is None else nlat.detach(), None if tlat is None else tlat.detach()))
        states = out; cache = {}; target_cache = {}

    rebuild(args.refresh)

    class H(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            u = urlparse(self.path)
            if u.path == "/":
                rows = []
                for sid, s in enumerate(states):
                    sigma_cells = "".join(
                        f"<div class='cell'><div class='lbl'>{_sigma_label(args, sg)}</div><audio controls preload='none' src='/audio?sid={sid}&kind=recon&sigma={sg:.6f}'></audio></div>"
                        for sg in sigma_variants
                    )
                    rows.append(
                        "<div class='row'>"
                        f"<div class='meta'><div class='k'>#{sid} idx={s.sample_index}</div><div class='f'>{s.sample_filename}</div></div>"
                        f"<div class='cell'><div class='lbl'>target</div><audio controls preload='none' src='/audio?sid={sid}&kind=target'></audio></div>"
                        f"<div class='cell'><div class='lbl'>recon (sigma=0.50)</div><audio controls preload='none' src='/audio?sid={sid}&kind=recon'></audio></div>"
                        f"{sigma_cells}</div>"
                    )
                html = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Recon Control</title>
<style>body{{font-family:Arial;margin:16px}}.mono{{font-family:Consolas;font-size:12px}}.toolbar{{display:flex;gap:8px;margin-bottom:12px}}.row{{display:grid;grid-template-columns:360px repeat({2+len(sigma_variants)},240px);gap:8px;align-items:start;border:1px solid #ddd;border-radius:8px;padding:8px;margin-bottom:8px;overflow-x:auto}}.meta{{font-size:12px;line-height:1.3}}.k{{font-weight:700}}.f{{color:#333;word-break:break-all}}.cell{{display:flex;flex-direction:column;gap:4px}}.lbl{{font-size:11px;color:#555}}audio{{width:220px}}</style></head>
<body><h3>Recon Control Live</h3><p class='mono'>module={args.module} | seed={used_seed} | recon=0.50 | sigma_variants={sigma_variants}</p><div class='toolbar'><a href='/'>Reload</a><a href='/refresh'>Refresh Samples</a></div>{''.join(rows)}</body></html>"""
                b = html.encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
            if u.path == "/refresh":
                with lock:
                    rebuild(True)
                self.send_response(302); self.send_header("Location", "/"); self.end_headers(); return
            if u.path != "/audio":
                self.send_error(404, "not found"); return
            q = parse_qs(u.query)
            sid = int(q.get("sid", ["0"])[0]); kind = q.get("kind", ["target"])[0]; sigma = float(q.get("sigma", ["0.5"])[0])
            with lock:
                if sid < 0 or sid >= len(states):
                    self.send_error(404, "bad sid"); return
                s = states[sid]
                if kind == "target":
                    if sid not in target_cache:
                        target_cache[sid] = _wav_bytes(s.target_waveform, args.sample_rate)
                    wb = target_cache[sid]
                else:
                    k = (sid, round(float(sigma), 6))
                    if k not in cache:
                        y = _render_with_module(
                            model=model,
                            sample=s,
                            sigma=float(sigma),
                            args=args,
                            transient_axis=taxis,
                            transient_basis=tbasis,
                        )
                        cache[k] = _wav_bytes(y, args.sample_rate)
                        if len(cache) > 256:
                            cache.pop(next(iter(cache.keys())), None)
                    wb = cache[k]
            self.send_response(200); self.send_header("Content-Type", "audio/wav"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(wb))); self.end_headers(); self.wfile.write(wb)

    print(f"[INFO] module={args.module} run_dir={run_dir}")
    print(f"[INFO] config={cfg}")
    print(f"[INFO] ckpt={ckpt}")
    print(f"[INFO] data_dir={data_dir}")
    print(f"[OK] live server: http://{args.host}:{args.port}/")
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
