# scripts/build_modal_features.py
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio
from tqdm import tqdm

from drumblender.utils.modal_analysis import CQTModalAnalysis


def stable_id(rel_path: str) -> str:
    # deterministic numeric-ish id from path
    h = hashlib.md5(rel_path.encode("utf-8")).hexdigest()
    return str(int(h[:12], 16))


def list_wavs(root: Path) -> List[Path]:
    wavs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]
    wavs.sort()
    return wavs


def make_pack_key(processed_root: Path, wav_path: Path) -> Tuple[str, str, str]:
    """
    processed/<TYPE>/<INSTRUMENT>/...wav
      -> type=TYPE, instrument=INSTRUMENT, pack=TYPE:INSTRUMENT
    """
    rel = wav_path.relative_to(processed_root)
    parts = rel.parts
    type_name = parts[0] if len(parts) >= 1 else "unknown"
    inst_name = parts[1] if len(parts) >= 2 else "unknown"
    pack = f"{type_name}:{inst_name}"
    return type_name, inst_name, pack


def make_splits_by_pack(pack_keys: List[str], seed: int, train=0.8, val=0.1) -> Dict[str, str]:
    uniq = sorted(set(pack_keys))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(uniq), generator=g).tolist()
    uniq = [uniq[i] for i in perm]

    n = len(uniq)
    n_train = int(n * train)
    n_val = int(n * val)

    train_set = set(uniq[:n_train])
    val_set = set(uniq[n_train : n_train + n_val])
    test_set = set(uniq[n_train + n_val :])

    out: Dict[str, str] = {}
    for k in uniq:
        if k in train_set:
            out[k] = "train"
        elif k in val_set:
            out[k] = "val"
        else:
            out[k] = "test"
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    # inputs
    ap.add_argument("--processed_root", type=str, default="/private/datasets/processed")

    # outputs (IMPORTANT): keep everything under /private/datasets
    ap.add_argument("--out_dir", type=str, default="/private/datasets/modal_features/processed_modal_flat")
    ap.add_argument("--meta_name", type=str, default="metadata.json")

    # modal params (match your cfg defaults)
    ap.add_argument("--sample_rate", type=int, default=48000)
    ap.add_argument("--num_modes", type=int, default=64)

    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--fmin", type=int, default=20)
    ap.add_argument("--n_bins", type=int, default=240)
    ap.add_argument("--bins_per_octave", type=int, default=24)
    ap.add_argument("--min_length", type=int, default=10)
    ap.add_argument("--threshold_db", type=float, default=-80.0)
    ap.add_argument("--diff_threshold", type=float, default=5.0)

    # behavior
    ap.add_argument("--seed", type=int, default=5152845)
    ap.add_argument("--max_files", type=int, default=0, help="0 = all files")

    # Failure handling to push toward 'fail=0'
    ap.add_argument("--pad_short", action="store_true", help="If CQT reflect-pad error, right-pad zeros and retry.")
    ap.add_argument("--pad_to", type=int, default=0, help="If >0, right-pad audio shorter than this to this length (samples).")
    ap.add_argument("--min_duration_ms", type=float, default=0.0, help="If >0, skip files shorter than this (ms). Set 0 to not skip.")
    ap.add_argument("--allow_multich", action="store_true", help="If set, keep multi-channel; otherwise use first channel.")
    args = ap.parse_args()

    processed_root = Path(args.processed_root)
    if not processed_root.exists():
        raise FileNotFoundError(processed_root)

    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    feat_dir = out_dir / "features"
    audio_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    wavs = list_wavs(processed_root)
    if args.max_files and args.max_files > 0:
        wavs = wavs[: args.max_files]

    print(f"[scan] {processed_root} -> {len(wavs)} wavs")
    if len(wavs) == 0:
        raise RuntimeError("No wavs found.")

    typed: List[Tuple[str, str, str]] = []
    packs: List[str] = []
    for p in wavs:
        type_name, inst_name, pack = make_pack_key(processed_root, p)
        typed.append((type_name, inst_name, pack))
        packs.append(pack)
    pack2split = make_splits_by_pack(packs, seed=args.seed)

    modal = CQTModalAnalysis(
        args.sample_rate,
        hop_length=args.hop_length,
        fmin=args.fmin,
        n_bins=args.n_bins,
        bins_per_octave=args.bins_per_octave,
        min_length=args.min_length,
        num_modes=args.num_modes,
        threshold=args.threshold_db,
        diff_threshold=args.diff_threshold,
    )

    meta: Dict[str, Dict] = {}
    failed = 0
    kept = 0

    def right_pad_to(w: torch.Tensor, target: int) -> torch.Tensor:
        T = int(w.shape[-1])
        if T >= target:
            return w
        return torch.nn.functional.pad(w, (0, target - T))

    def extract_feat(w: torch.Tensor) -> torch.Tensor:
        modal_freqs, modal_amps, modal_phases = modal(w)  # (1, M, F)
        modal_freqs = 2 * torch.pi * modal_freqs / args.sample_rate
        feat = torch.stack([modal_freqs, modal_amps, modal_phases])  # (3,1,M,F)
        feat = feat.squeeze(1)  # (3,M,F)
        return feat

    pbar = tqdm(list(zip(wavs, typed)), total=len(wavs), desc="modal", unit="file", dynamic_ncols=True)

    for wav_path, (type_name, inst_name, pack) in pbar:
        rel = wav_path.relative_to(processed_root)
        key = stable_id(str(rel))

        try:
            wav, sr = torchaudio.load(str(wav_path))  # [C,T]
            if wav.ndim != 2:
                raise ValueError(f"bad wav shape: {tuple(wav.shape)}")

            # keep your preprocessing result as much as possible:
            # - DO NOT resample here. Just assert.
            if sr != args.sample_rate:
                raise ValueError(f"sample_rate mismatch: {sr} != {args.sample_rate} for {rel}")

            # channel handling: default keep first channel (no downmix)
            if (not args.allow_multich) and wav.shape[0] > 1:
                wav = wav[:1, :]
            elif wav.shape[0] == 0:
                raise ValueError("empty channel dim")

            # optional hard skip for too-short (you can keep 0.0 to not skip)
            if args.min_duration_ms and args.min_duration_ms > 0:
                min_samples = int(args.sample_rate * (args.min_duration_ms / 1000.0))
                if wav.shape[-1] < min_samples:
                    raise ValueError(f"too_short: {wav.shape[-1]} < {min_samples} samples")

            # optional fixed padding floor
            if args.pad_to and args.pad_to > 0:
                wav = right_pad_to(wav, args.pad_to)

            # extract with retry for reflect padding error
            try:
                feat = extract_feat(wav)
            except RuntimeError as e:
                if args.pad_short and "Padding size should be less than the corresponding input dimension" in str(e):
                    m = re.search(r"padding\s+\((\d+),\s*(\d+)\)", str(e))
                    if m:
                        pad_l = int(m.group(1))
                        target = pad_l + 1
                        wav2 = right_pad_to(wav, target)
                        feat = extract_feat(wav2)
                    else:
                        raise
                else:
                    raise

            # force fixed num_modes
            P, M, F = feat.shape
            if M < args.num_modes:
                pad = feat.new_zeros((P, args.num_modes - M, F))
                feat = torch.cat([feat, pad], dim=1)
            elif M > args.num_modes:
                feat = feat[:, : args.num_modes, :]

            # save outputs under out_dir
            out_wav = audio_dir / f"{key}.wav"
            out_feat = feat_dir / f"{key}.pt"
            torchaudio.save(str(out_wav), wav, args.sample_rate)
            torch.save(feat, out_feat)

            split = pack2split[pack]
            meta[key] = {
                "filename": str(out_wav.relative_to(out_dir)),
                "feature_file": str(out_feat.relative_to(out_dir)),
                "sample_pack_key": pack,
                "instrument": inst_name,
                "type": type_name,
                "split": split,
                "num_samples": int(wav.shape[-1]),
                "orig_relpath": str(rel),
            }

            kept += 1

        except Exception as e:
            failed += 1
            # keep moving; user will run stats and we’ll fix root cause
            # write a minimal failure record into meta as well (optional)
            # (we keep it out of metadata.json so dataset doesn't break)
            # print is fine for now
            print("fail:", str(rel), "->", repr(e))

        pbar.set_postfix(kept=kept, failed=failed)

    meta_path = out_dir / args.meta_name
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print("[done] kept:", len(meta), "failed:", failed)
    print("out_dir:", out_dir)
    print("meta:", meta_path)


if __name__ == "__main__":
    main()
