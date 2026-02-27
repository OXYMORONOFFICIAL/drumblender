#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchaudio
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from drumblender.data.audio import AudioWithParametersDataset
from drumblender.utils.model import load_model


DEFAULT_SEED = 20260218
DEFAULT_SR = 48000
DEFAULT_DATA_DIR = "/workspace/datasets/modal_features/processed_modal_flat"


@dataclass
class AxisCandidate:
    index: int
    eigval: float
    step: float
    change_mean: float
    fail_clip_ratio: float
    fail_hf_ratio: float
    fail_env_ratio: float
    passed: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _latest(paths: Sequence[Path]) -> Optional[Path]:
    files = [p for p in paths if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _device(v: str) -> torch.device:
    if v == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(v)


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
        raise FileNotFoundError(f"--ckpt not found: {p}")
    p = _latest(list(run_dir.rglob("*.ckpt")))
    if p is None:
        raise FileNotFoundError("Could not infer checkpoint in run_dir.")
    return p.resolve()


def _resolve_encoder_cfg(
    cfg_dir: Path, kind: str, backbone: str, explicit_cfg: Optional[str], root: Path
) -> Optional[str]:
    if explicit_cfg:
        p = Path(explicit_cfg)
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


def _build_resolved_config(
    base_cfg: Path,
    loss_cfg: Optional[str],
    noise_backbone: str,
    transient_backbone: str,
    noise_cfg: Optional[str],
    transient_cfg: Optional[str],
    root: Path,
) -> Tuple[Path, bool]:
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
        p = Path(loss_cfg)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--loss-cfg not found: {p}")
        init_args["loss_fn"] = str(p)

    ncfg = _resolve_encoder_cfg(cfg_dir, "noise", noise_backbone, noise_cfg, root)
    if ncfg:
        init_args["noise_autoencoder"] = ncfg
        init_args["noise_autoencoder_accepts_audio"] = True

    tcfg = _resolve_encoder_cfg(cfg_dir, "transient", transient_backbone, transient_cfg, root)
    if tcfg:
        init_args["transient_autoencoder"] = tcfg
        init_args["transient_autoencoder_accepts_audio"] = True

    tmp = cfg_dir / f".effect_axes_resolved_{int(time.time())}.yaml"
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return tmp, True


def _unwrap(x):
    return x[0] if isinstance(x, tuple) else x


@torch.no_grad()
def _compute_latents(model, x: torch.Tensor, p: torch.Tensor):
    emb = None if model.encoder is None else model.encoder(x)
    modal = p
    if model.modal_autoencoder is not None:
        if model.modal_autoencoder_accepts_audio:
            modal = _unwrap(model.modal_autoencoder(x, p))
        else:
            modal = _unwrap(model.modal_autoencoder(emb, p))

    noise_lat = None
    if model.noise_autoencoder is not None:
        if model.noise_autoencoder_accepts_audio:
            noise_lat = _unwrap(model.noise_autoencoder(x))
        else:
            noise_lat = _unwrap(model.noise_autoencoder(emb))

    transient_lat = None
    if model.transient_autoencoder is not None:
        if model.transient_autoencoder_accepts_audio:
            transient_lat = _unwrap(model.transient_autoencoder(x))
        else:
            transient_lat = _unwrap(model.transient_autoencoder(emb))

    return modal, noise_lat, transient_lat


@torch.no_grad()
def _prepare_transient_inputs(
    model, waveform: torch.Tensor, params: torch.Tensor, length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      y_in: signal fed to transient_synth (shape [1,1,L])
      z0: transient latent (shape [1,D])
    """
    x = waveform.unsqueeze(0)
    p = params.unsqueeze(0)
    modal_params, noise_lat, transient_lat = _compute_latents(model, x, p)
    if transient_lat is None:
        raise RuntimeError("Model has no transient_autoencoder output.")
    if transient_lat.ndim != 2:
        raise RuntimeError(
            f"Expected transient latent [B,D], got {tuple(transient_lat.shape)}."
        )

    y_in = model.modal_synth(modal_params, int(length))
    if model.noise_synth is not None and model.transient_takes_noise:
        if noise_lat is None:
            raise RuntimeError("transient_takes_noise=True but noise latent is missing.")
        try:
            n = model.noise_synth(noise_lat, int(length)).unsqueeze(1)
        except Exception:
            n = model.noise_synth(noise_lat.transpose(1, 2), int(length)).unsqueeze(1)
        y_in = y_in + n

    return y_in, transient_lat


def _parse_mrstft_specs(spec_str: str) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for token in spec_str.split(","):
        t = token.strip()
        if not t:
            continue
        parts = t.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --mrstft-specs token '{t}'. Expected fft:hop:win."
            )
        out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    if not out:
        raise ValueError("No valid MR-STFT specs parsed.")
    return out


def _norm_block(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = v.mean(dim=-1, keepdim=True)
    s = v.std(dim=-1, keepdim=True, unbiased=False)
    return (v - m) / (s + eps)


def _estimate_onset_idx(wave: torch.Tensor, frac: float) -> int:
    """
    wave: [1, T]
    """
    x = wave.abs().squeeze(0)
    if x.numel() < 2:
        return 0
    peak = float(x.max().item())
    if peak <= 1e-8:
        return 0
    thr = max(1e-8, frac * peak)
    idx = (x >= thr).nonzero(as_tuple=False)
    return int(idx[0, 0].item()) if idx.numel() > 0 else 0


def _crop_with_pad(x: torch.Tensor, start: int, size: int) -> torch.Tensor:
    """
    x: [B,1,T]
    """
    T = x.shape[-1]
    s = max(0, min(int(start), max(0, T - 1)))
    e = s + int(size)
    if e <= T:
        return x[..., s:e]
    pad = e - T
    return F.pad(x[..., s:T], (0, pad))


class PhiExtractor:
    def __init__(
        self,
        sample_rate: int,
        mel_n_fft: int,
        mel_hop: int,
        mel_win: int,
        mel_bins: int,
        mr_specs: List[Tuple[int, int, int]],
        mr_bins: int,
        eps: float,
        l2_norm: bool,
        device: torch.device,
    ):
        self.sample_rate = int(sample_rate)
        self.mel_bins = int(mel_bins)
        self.mr_specs = list(mr_specs)
        self.mr_bins = int(mr_bins)
        self.eps = float(eps)
        self.l2_norm = bool(l2_norm)
        self.device = device

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=int(mel_n_fft),
            hop_length=int(mel_hop),
            win_length=int(mel_win),
            n_mels=self.mel_bins,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        ).to(device)
        self._win_cache: Dict[int, torch.Tensor] = {}

    def _window(self, n_fft: int, dtype: torch.dtype) -> torch.Tensor:
        key = int(n_fft)
        w = self._win_cache.get(key, None)
        if w is None or w.device != self.device or w.dtype != dtype:
            w = torch.hann_window(key, device=self.device, dtype=dtype)
            self._win_cache[key] = w
        return w

    def __call__(self, tw: torch.Tensor) -> torch.Tensor:
        """
        tw: [B, 1, T]
        returns phi: [B, Dphi]
        """
        wav = tw.squeeze(1)
        blocks: List[torch.Tensor] = []

        mel = self.mel(wav).clamp_min(self.eps).log()
        mel = mel.mean(dim=-1)  # [B, mel_bins]
        mel = _norm_block(mel, eps=self.eps)
        blocks.append(mel)

        for n_fft, hop, win in self.mr_specs:
            stft = torch.stft(
                wav,
                n_fft=int(n_fft),
                hop_length=int(hop),
                win_length=int(win),
                window=self._window(int(n_fft), wav.dtype),
                center=True,
                return_complex=True,
            )
            mag = stft.abs().clamp_min(self.eps).log()
            feat = mag.mean(dim=-1)  # [B, F]
            if self.mr_bins > 0 and feat.shape[-1] != self.mr_bins:
                feat = F.interpolate(
                    feat.unsqueeze(1),
                    size=self.mr_bins,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
            feat = _norm_block(feat, eps=self.eps)
            blocks.append(feat)

        phi = torch.cat(blocks, dim=-1)
        if self.l2_norm:
            phi = phi / (phi.norm(dim=-1, keepdim=True) + self.eps)
        return phi


def _rademacher_like(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape).bernoulli_(0.5).mul_(2.0).sub_(1.0)


def _step_from_lambda(lam: float, target: float, eps: float) -> float:
    return float(target / math.sqrt(max(float(lam), 0.0) + eps))


def _clamp_z_stats(z: torch.Tensor, z_mean: torch.Tensor, z_std: torch.Tensor, q: float):
    zn = (z - z_mean) / (z_std + 1e-8)
    zn = torch.clamp(zn, -float(q), float(q))
    return z_mean + zn * z_std


def _rms_match(x: torch.Tensor, ref: torch.Tensor, clamp_min: float, clamp_max: float):
    xr = torch.sqrt(torch.mean(x * x) + 1e-8)
    rr = torch.sqrt(torch.mean(ref * ref) + 1e-8)
    g = torch.clamp(rr / xr, float(clamp_min), float(clamp_max))
    return x * g


def _hf_ratio(x: torch.Tensor, sample_rate: int, n_fft: int, hop: int, win: int, hf_cut_hz: float):
    wav = x.squeeze(1)
    stft = torch.stft(
        wav,
        n_fft=int(n_fft),
        hop_length=int(hop),
        win_length=int(win),
        window=torch.hann_window(int(n_fft), device=x.device, dtype=x.dtype),
        center=True,
        return_complex=True,
    )
    p = stft.abs().pow(2).mean(dim=-1)  # [B, F]
    freqs = torch.linspace(0.0, sample_rate / 2.0, p.shape[-1], device=x.device)
    hf_mask = (freqs >= float(hf_cut_hz)).float()[None, :]
    hf = (p * hf_mask).sum(dim=-1)
    allp = p.sum(dim=-1).clamp_min(1e-8)
    return (hf / allp).mean()


def _env_diff(x: torch.Tensor, ref: torch.Tensor, sample_rate: int, win_ms: float):
    k = max(1, int(round(float(sample_rate) * float(win_ms) / 1000.0)))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xe = F.avg_pool1d(x.abs(), kernel_size=k, stride=1, padding=pad)
    re = F.avg_pool1d(ref.abs(), kernel_size=k, stride=1, padding=pad)
    return torch.mean(torch.abs(xe - re))


def _select_indices(n_total: int, n_pick: int, seed: int) -> List[int]:
    if n_pick >= n_total:
        return list(range(n_total))
    rng = random.Random(seed)
    return rng.sample(list(range(n_total)), int(n_pick))


def _save_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract z_global effect axes via Hutchinson E[J^T J] for transient control."
    )
    p.add_argument("run_dir", type=str, help="Run directory containing checkpoints/context.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--meta-file", type=str, default="metadata.json")
    p.add_argument("--split-strategy", type=str, default="sample_pack", choices=["sample_pack", "random"])
    p.add_argument("--parameter-key", type=str, default="feature_file")
    p.add_argument("--expected-num-modes", type=int, default=64)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    p.add_argument("--loss-cfg", type=str, default=None)
    p.add_argument("--noise-encoder-backbone", type=str, default=None)
    p.add_argument("--transient-encoder-backbone", type=str, default=None)
    p.add_argument("--noise-encoder-cfg", type=str, default=None)
    p.add_argument("--transient-encoder-cfg", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    p.add_argument("--subset-size", type=int, default=3000)
    p.add_argument("--holdout-size", type=int, default=256)
    p.add_argument("--hutch-probes", type=int, default=8)
    p.add_argument("--candidate-k", type=int, default=8)
    p.add_argument("--final-k", type=int, default=2)
    p.add_argument("--step-target", type=float, default=1.0)
    p.add_argument("--z-clamp-q", type=float, default=2.5)

    p.add_argument("--window-ms", type=float, default=120.0)
    p.add_argument("--onset-threshold-frac", type=float, default=0.12)

    p.add_argument("--mel-n-fft", type=int, default=1024)
    p.add_argument("--mel-hop", type=int, default=256)
    p.add_argument("--mel-win", type=int, default=1024)
    p.add_argument("--mel-bins", type=int, default=96)
    p.add_argument("--mrstft-specs", type=str, default="1024:256:1024,2048:512:2048,512:128:512")
    p.add_argument("--mrstft-bins", type=int, default=96)
    p.add_argument("--phi-l2-norm", action="store_true")

    p.add_argument("--sweep-points", type=int, default=9)
    p.add_argument("--clip-threshold", type=float, default=0.99)
    p.add_argument("--clip-rate-threshold", type=float, default=0.005)
    p.add_argument("--max-fail-clip-ratio", type=float, default=0.05)
    p.add_argument("--hf-cut-hz", type=float, default=10000.0)
    p.add_argument("--hf-delta-threshold", type=float, default=0.08)
    p.add_argument("--max-fail-hf-ratio", type=float, default=0.15)
    p.add_argument("--env-win-ms", type=float, default=10.0)
    p.add_argument("--env-diff-threshold", type=float, default=0.12)
    p.add_argument("--max-fail-env-ratio", type=float, default=0.20)
    p.add_argument("--rms-match-min", type=float, default=0.5)
    p.add_argument("--rms-match-max", type=float, default=2.0)

    p.add_argument("--shrinkage-a", type=float, default=0.02)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save axes (default: <run_dir>/analysis/effect_axes_<timestamp>)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = _repo_root()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    ctx = _read_context(run_dir)
    cfg = _infer_config(run_dir, args.config, ctx, root)
    ckpt = _infer_ckpt(run_dir, args.ckpt)
    data_dir = Path(
        args.data_dir or _localize_path(ctx.get("data_dir"), root) or DEFAULT_DATA_DIR
    ).expanduser().resolve()

    noise_backbone = args.noise_encoder_backbone or str(
        ctx.get("noise_encoder_backbone") or "soundstream"
    )
    transient_backbone = args.transient_encoder_backbone or str(
        ctx.get("transient_encoder_backbone") or "soundstream"
    )

    resolved_cfg, is_tmp = _build_resolved_config(
        base_cfg=cfg,
        loss_cfg=args.loss_cfg,
        noise_backbone=noise_backbone,
        transient_backbone=transient_backbone,
        noise_cfg=args.noise_encoder_cfg,
        transient_cfg=args.transient_encoder_cfg,
        root=root,
    )

    try:
        model, _ = load_model(str(resolved_cfg), str(ckpt), include_data=False)
    finally:
        if is_tmp and resolved_cfg.exists():
            resolved_cfg.unlink(missing_ok=True)

    dev = _device(args.device)
    model.eval().to(dev)

    dataset = AudioWithParametersDataset(
        data_dir=str(data_dir),
        meta_file=args.meta_file,
        sample_rate=int(args.sample_rate),
        num_samples=None,
        split="train",
        split_strategy=args.split_strategy,
        parameter_key=args.parameter_key,
        expected_num_modes=args.expected_num_modes,
        seed=int(args.seed),
    )

    if model.transient_synth is None:
        raise RuntimeError("Model has no transient_synth.")
    if model.transient_autoencoder is None:
        raise RuntimeError("Model has no transient_autoencoder.")

    mr_specs = _parse_mrstft_specs(args.mrstft_specs)
    phi_fn = PhiExtractor(
        sample_rate=int(args.sample_rate),
        mel_n_fft=int(args.mel_n_fft),
        mel_hop=int(args.mel_hop),
        mel_win=int(args.mel_win),
        mel_bins=int(args.mel_bins),
        mr_specs=mr_specs,
        mr_bins=int(args.mrstft_bins),
        eps=1e-8,
        l2_norm=bool(args.phi_l2_norm),
        device=dev,
    )

    subset_indices = _select_indices(len(dataset), int(args.subset_size), int(args.seed))
    holdout_seed = int(args.seed) + 1337
    holdout_indices = _select_indices(
        len(dataset), int(args.holdout_size), holdout_seed
    )

    print(f"[INFO] train subset: {len(subset_indices)} | holdout: {len(holdout_indices)}")
    print(f"[INFO] device: {dev} | cfg: {cfg} | ckpt: {ckpt}")

    # Step 1: collect z stats.
    z_list: List[torch.Tensor] = []
    for idx in tqdm(subset_indices, desc="collect_z"):
        waveform, params, length = dataset[idx]
        waveform = waveform.to(dev)
        params = params.to(dev)
        with torch.no_grad():
            _, z0 = _prepare_transient_inputs(
                model=model, waveform=waveform, params=params, length=int(length)
            )
        z_list.append(z0.squeeze(0).detach().cpu())

    Z = torch.stack(z_list, dim=0)  # [N, D]
    z_mean = Z.mean(dim=0).to(dev)
    z_std = Z.std(dim=0, unbiased=False).clamp_min(1e-8).to(dev)
    z_dim = int(Z.shape[-1])
    del Z, z_list

    # Step 2: estimate M = E[g g^T], g = normalized(J^T r)
    M = torch.zeros(z_dim, z_dim, device=dev)
    probes = int(args.hutch_probes)

    for idx in tqdm(subset_indices, desc="estimate_M"):
        waveform, params, length = dataset[idx]
        waveform = waveform.to(dev)
        params = params.to(dev)

        with torch.no_grad():
            y_in, z0 = _prepare_transient_inputs(
                model=model, waveform=waveform, params=params, length=int(length)
            )
            onset = _estimate_onset_idx(waveform, frac=float(args.onset_threshold_frac))
            win = max(16, int(round(float(args.window_ms) * float(args.sample_rate) / 1000.0)))

        z = z0.detach().requires_grad_(True)
        t = model.transient_synth(y_in, z)
        tw = _crop_with_pad(t, start=onset, size=win)
        phi = phi_fn(tw)  # [1, Dphi]

        for ri in range(probes):
            r = _rademacher_like(phi)
            s = (phi * r).sum()
            g = torch.autograd.grad(
                s,
                z,
                retain_graph=(ri < probes - 1),
                create_graph=False,
                allow_unused=False,
            )[0]  # [1, D]
            g = g / (g.norm(dim=-1, keepdim=True) + 1e-8)
            gv = g.squeeze(0).detach()
            M += torch.outer(gv, gv)

    M /= max(1, len(subset_indices) * probes)
    a = float(args.shrinkage_a)
    if a > 0:
        I = torch.eye(z_dim, device=dev, dtype=M.dtype)
        M = (1.0 - a) * M + a * I

    evals, evecs = torch.linalg.eigh(M)
    order = torch.argsort(evals, descending=True)
    kcand = max(1, min(int(args.candidate_k), z_dim))
    cand_idx = order[:kcand]
    cand_evals = evals[cand_idx]
    cand_axes = evecs[:, cand_idx]  # [D, K]

    # Step 3: candidate sweep and safety filtering.
    t_grid = torch.linspace(-1.0, 1.0, steps=max(3, int(args.sweep_points)), device=dev)
    candidates: List[AxisCandidate] = []

    with torch.no_grad():
        for k in range(kcand):
            u = cand_axes[:, k]
            lam = float(cand_evals[k].item())
            step = _step_from_lambda(
                lam=lam, target=float(args.step_target), eps=1e-8
            )

            fail_clip = 0
            fail_hf = 0
            fail_env = 0
            total = 0
            change_accum = 0.0

            for idx in holdout_indices:
                waveform, params, length = dataset[idx]
                waveform = waveform.to(dev)
                params = params.to(dev)
                y_in, z0 = _prepare_transient_inputs(
                    model=model, waveform=waveform, params=params, length=int(length)
                )

                onset = _estimate_onset_idx(waveform, frac=float(args.onset_threshold_frac))
                win = max(
                    16,
                    int(round(float(args.window_ms) * float(args.sample_rate) / 1000.0)),
                )

                t0 = model.transient_synth(y_in, z0)
                tw0 = _crop_with_pad(t0, start=onset, size=win)
                phi0 = phi_fn(tw0)
                hf0 = _hf_ratio(
                    tw0,
                    sample_rate=int(args.sample_rate),
                    n_fft=int(args.mel_n_fft),
                    hop=int(args.mel_hop),
                    win=int(args.mel_win),
                    hf_cut_hz=float(args.hf_cut_hz),
                )

                for tv in t_grid:
                    z1 = z0 + (float(tv.item()) * step) * u.unsqueeze(0)
                    z1 = _clamp_z_stats(
                        z1, z_mean=z_mean.unsqueeze(0), z_std=z_std.unsqueeze(0), q=float(args.z_clamp_q)
                    )
                    t1 = model.transient_synth(y_in, z1)
                    tw1 = _crop_with_pad(t1, start=onset, size=win)
                    tw1 = _rms_match(
                        tw1,
                        tw0,
                        clamp_min=float(args.rms_match_min),
                        clamp_max=float(args.rms_match_max),
                    )

                    clip_rate = float((tw1.abs() > float(args.clip_threshold)).float().mean().item())
                    hf1 = _hf_ratio(
                        tw1,
                        sample_rate=int(args.sample_rate),
                        n_fft=int(args.mel_n_fft),
                        hop=int(args.mel_hop),
                        win=int(args.mel_win),
                        hf_cut_hz=float(args.hf_cut_hz),
                    )
                    envd = float(
                        _env_diff(
                            tw1,
                            tw0,
                            sample_rate=int(args.sample_rate),
                            win_ms=float(args.env_win_ms),
                        ).item()
                    )
                    phi1 = phi_fn(tw1)
                    ch = float(torch.norm(phi1 - phi0, dim=-1).mean().item())

                    change_accum += ch
                    total += 1
                    if clip_rate > float(args.clip_rate_threshold):
                        fail_clip += 1
                    if float(hf1.item() - hf0.item()) > float(args.hf_delta_threshold):
                        fail_hf += 1
                    if envd > float(args.env_diff_threshold):
                        fail_env += 1

            total = max(total, 1)
            fail_clip_ratio = fail_clip / total
            fail_hf_ratio = fail_hf / total
            fail_env_ratio = fail_env / total
            passed = (
                fail_clip_ratio <= float(args.max_fail_clip_ratio)
                and fail_hf_ratio <= float(args.max_fail_hf_ratio)
                and fail_env_ratio <= float(args.max_fail_env_ratio)
            )

            candidates.append(
                AxisCandidate(
                    index=k,
                    eigval=lam,
                    step=step,
                    change_mean=change_accum / total,
                    fail_clip_ratio=fail_clip_ratio,
                    fail_hf_ratio=fail_hf_ratio,
                    fail_env_ratio=fail_env_ratio,
                    passed=passed,
                )
            )

    passed = [c for c in candidates if c.passed]
    passed = sorted(passed, key=lambda c: c.change_mean, reverse=True)
    if len(passed) == 0:
        print("[WARN] no axis passed safety; fallback to highest-change candidates.")
        fallback = sorted(candidates, key=lambda c: c.change_mean, reverse=True)
        picked = fallback[: max(1, int(args.final_k))]
    else:
        picked = passed[: max(1, int(args.final_k))]

    final_indices = [p.index for p in picked]
    final_axes = cand_axes[:, final_indices].T.detach().cpu()  # [K,D]
    final_steps = torch.tensor([p.step for p in picked], dtype=torch.float32)
    final_evals = torch.tensor([p.eigval for p in picked], dtype=torch.float32)

    # Save.
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (run_dir / "analysis" / f"effect_axes_{ts}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "axes": final_axes,
        "steps": final_steps,
        "eigvals": final_evals,
        "z_mean": z_mean.detach().cpu(),
        "z_std": z_std.detach().cpu(),
        "candidate_axes": cand_axes.T.detach().cpu(),
        "candidate_eigvals": cand_evals.detach().cpu(),
        "candidate_report": [c.__dict__ for c in candidates],
        "picked_candidate_indices": final_indices,
        "mrstft_specs": mr_specs,
        "cfg_path": str(cfg),
        "ckpt_path": str(ckpt),
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "args": vars(args),
    }

    pt_path = out_dir / "effect_axes.pt"
    torch.save(payload, pt_path)
    _save_json(
        out_dir / "summary.json",
        {
            "picked_indices": final_indices,
            "picked_steps": [float(x) for x in final_steps.tolist()],
            "picked_eigvals": [float(x) for x in final_evals.tolist()],
            "n_candidates": len(candidates),
            "n_passed": len(passed),
            "subset_size": len(subset_indices),
            "holdout_size": len(holdout_indices),
            "hutch_probes": int(args.hutch_probes),
            "output_pt": str(pt_path),
        },
    )

    if ctx:
        _save_json(out_dir / "run_context_used.json", ctx)

    print(f"[OK] saved: {pt_path}")
    print(f"[OK] summary: {out_dir / 'summary.json'}")
    print("[INFO] picked axes:")
    for i, p in enumerate(picked):
        print(
            f"  axis#{i} <- cand#{p.index} | eig={p.eigval:.6g} | step={p.step:.6g} "
            f"| change={p.change_mean:.6g} | fail(clip/hf/env)="
            f"{p.fail_clip_ratio:.3f}/{p.fail_hf_ratio:.3f}/{p.fail_env_ratio:.3f}"
        )


if __name__ == "__main__":
    main()
