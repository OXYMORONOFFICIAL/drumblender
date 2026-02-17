from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
from tqdm import tqdm
import math

from drumblender.utils.modal_analysis import CQTModalAnalysis


def stable_id(rel_path: str) -> str:
    # deterministic numeric-ish id from path
    h = hashlib.md5(rel_path.encode("utf-8")).hexdigest()
    return str(int(h[:12], 16))


def list_wavs(root: Path) -> List[Path]:
    wavs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]
    wavs.sort()
    return wavs


def make_pack_key(
    processed_root: Path,
    wav_path: Path,
    pack_depth: int = 1,
) -> Tuple[str, str, str]:
    """
    Build a pack key directly from the top-level folder(s) under processed_root.

    Example:
      processed/pack_1/a.wav            -> pack=pack_1
      processed/pack_1/sub/x.wav        -> pack=pack_1
      processed/pack_2/sub/deep/y.wav   -> pack=pack_2

    If pack_depth > 1, the first N directory levels are joined.
    This still ignores deeper subdirectory structure for split grouping.
    """
    rel = wav_path.relative_to(processed_root)
    parts = rel.parts

    # HIGHLIGHT: Custom dataset policy.
    # We treat the top-level folder under processed_root as the pack id.
    # Subdirectories under the pack are ignored for pack grouping.
    type_name = "custom"
    inst_name = "unlabeled"

    inner_dirs = list(parts[:-1])  # all directories before filename
    depth = max(1, int(pack_depth))
    if len(inner_dirs) == 0:
        pack_name = "__root__"
    else:
        pack_name = "/".join(inner_dirs[:depth])

    pack = pack_name
    return type_name, inst_name, pack


def make_splits_within_pack(
    pack_keys: List[str],
    seed: int,
    train: float = 0.8,
    val: float = 0.1,
) -> List[str]:
    """
    Split files within each pack.
    """
    if train < 0.0 or val < 0.0 or (train + val) > 1.0:
        raise ValueError("Invalid split ratios: require train >= 0, val >= 0, train+val <= 1")

    by_pack: Dict[str, List[int]] = {}
    for idx, pack in enumerate(pack_keys):
        by_pack.setdefault(pack, []).append(idx)

    g = torch.Generator().manual_seed(seed)
    out = ["train"] * len(pack_keys)

    for pack in sorted(by_pack.keys()):
        idxs = by_pack[pack]
        perm = torch.randperm(len(idxs), generator=g).tolist()
        shuffled = [idxs[i] for i in perm]

        n = len(shuffled)
        n_train = int(n * train)
        n_val = int(n * val)

        # Ensure each pack contributes at least one training sample where possible.
        if n > 0 and n_train == 0:
            n_train = 1
        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        for rank, original_idx in enumerate(shuffled):
            if rank < n_train:
                out[original_idx] = "train"
            elif rank < (n_train + n_val):
                out[original_idx] = "val"
            else:
                out[original_idx] = "test"

    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    # inputs
    ap.add_argument("--processed_root", type=str, default="/private/datasets/processed")

    # outputs
    ap.add_argument("--out_dir", type=str, default="/private/datasets/modal_features/processed_modal_flat")
    ap.add_argument("--meta_name", type=str, default="metadata.json")

    # modal params
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
    ap.add_argument(
        "--pack_depth",
        type=int,
        default=1,
        help=(
            "Number of top-level path segments under processed_root used as pack id. "
            "1 means top-level folder only (e.g., processed/pack_name/...)."
        ),
    )
    ap.add_argument(
        "--write_split",
        action="store_true",
        help=(
            "Write split labels into metadata during preprocessing. "
            "Default is OFF so split is done at training time in the dataset."
        ),
    )

    # failure handling
    # HIGHLIGHT: Auto-padding retry is enabled by default to handle short files
    # that fail inside nnAudio CQT reflect padding.
    ap.add_argument(
        "--pad_short",
        dest="pad_short",
        action="store_true",
        help="Enable auto right-padding retry for CQT reflect-pad failures (default: enabled).",
    )
    ap.add_argument(
        "--no_pad_short",
        dest="pad_short",
        action="store_false",
        help="Disable auto right-padding retry for CQT reflect-pad failures.",
    )
    ap.set_defaults(pad_short=True)
    ap.add_argument("--pad_to", type=int, default=0, help="If >0, right-pad audio shorter than this to this length (samples).")
    ap.add_argument("--min_duration_ms", type=float, default=0.0, help="If >0, skip files shorter than this (ms). Set 0 to not skip.")
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
        type_name, inst_name, pack = make_pack_key(
            processed_root,
            p,
            pack_depth=args.pack_depth,
        )
        typed.append((type_name, inst_name, pack))
        packs.append(pack)

    split_by_index = None
    if args.write_split:
        # HIGHLIGHT: Optional backward compatibility path.
        # For the current workflow we keep this disabled so the dataset class
        # computes split dynamically from sample_pack_key and seed.
        split_by_index = make_splits_within_pack(packs, seed=args.seed)

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
        t = int(w.shape[-1])
        if t >= target:
            return w
        return torch.nn.functional.pad(w, (0, target - t))

    def num_frames_from_len(T: int, hop: int) -> int:
        # 대충 CQT 결과 프레임 수와 비슷하게 맞추는 용도 (최소 1)
        return max(1, math.ceil(T / hop))

    def extract_feat(w: torch.Tensor) -> torch.Tensor:
        """
        returns feat: (3, M, F)  where M can be 0..num_modes
        if M==0, return zeros (3, 0, F) instead of crashing.
        """
        try:
            modal_freqs, modal_amps, modal_phases = modal(w)  # expected (1, M, F)
        except RuntimeError as e:
            msg = str(e)
            # 0개 모달 케이스가 내부에서 torch.stack([])로 터지는 경우를 흡수
            if "non-empty TensorList" in msg:
                F = num_frames_from_len(int(w.shape[-1]), args.hop_length)
                z = w.new_zeros((3, 0, F))
                return z
            raise

        # 혹시라도 구현이 M=0 텐서를 반환하는 경우까지 방어
        if modal_freqs.numel() == 0 or modal_freqs.shape[1] == 0:
            F = num_frames_from_len(int(w.shape[-1]), args.hop_length)
            return w.new_zeros((3, 0, F))

        modal_freqs = 2 * torch.pi * modal_freqs / args.sample_rate
        feat = torch.stack([modal_freqs, modal_amps, modal_phases])  # (3,1,M,F)
        feat = feat.squeeze(1)  # (3,M,F)
        return feat

    def infer_required_length_from_padding_error(msg: str, current_len: int) -> int:
        # HIGHLIGHT: Parse nnAudio/torch padding error and choose a safe retry length.
        m = re.search(r"padding\s+\((\d+),\s*(\d+)\)", msg)
        if m:
            pad_l = int(m.group(1))
            pad_r = int(m.group(2))
            return max(current_len, pad_l + 1, pad_r + 1)

        # Fallback if the exact tuple is missing from the error string.
        return max(current_len + 1, current_len * 2)

    pbar = tqdm(
        list(zip(wavs, typed)),
        total=len(wavs),
        desc="modal",
        unit="file",
        dynamic_ncols=True,
    )

    for idx, (wav_path, (type_name, inst_name, pack)) in enumerate(pbar):
        rel = wav_path.relative_to(processed_root)
        key = stable_id(str(rel))

        try:
            wav, sr = torchaudio.load(str(wav_path))  # [C,T]
            if wav.ndim != 2:
                raise ValueError(f"bad wav shape: {tuple(wav.shape)}")

            # no resample: enforce preprocessed sample rate
            if sr != args.sample_rate:
                raise ValueError(f"sample_rate mismatch: {sr} != {args.sample_rate} for {rel}")

            # mono-only policy: always channel 0
            if wav.shape[0] == 0:
                raise ValueError("empty channel dim")
            wav = wav[:1, :]

            # optional hard skip for too-short
            if args.min_duration_ms and args.min_duration_ms > 0:
                min_samples = int(args.sample_rate * (args.min_duration_ms / 1000.0))
                if wav.shape[-1] < min_samples:
                    raise ValueError(f"too_short: {wav.shape[-1]} < {min_samples} samples")

            # optional fixed padding floor
            if args.pad_to and args.pad_to > 0:
                wav = right_pad_to(wav, args.pad_to)

            # extract with retry for reflect padding errors from CQT
            try:
                feat = extract_feat(wav)
            except RuntimeError as e:
                msg = str(e)
                if (not args.pad_short) or ("Padding size should be less than the corresponding input dimension" not in msg):
                    raise

                # HIGHLIGHT: Single retry only.
                # Apply one right-padding pass and retry once; if it still fails,
                # propagate the error and mark this file as failed.
                target = infer_required_length_from_padding_error(msg, int(wav.shape[-1]))
                wav_try = right_pad_to(wav, target)
                feat = extract_feat(wav_try)

            # force fixed num_modes
            p, m, f = feat.shape
            if m < args.num_modes:
                pad = feat.new_zeros((p, args.num_modes - m, f))
                feat = torch.cat([feat, pad], dim=1)
            elif m > args.num_modes:
                feat = feat[:, : args.num_modes, :]

            # save outputs under out_dir
            out_wav = audio_dir / f"{key}.wav"
            out_feat = feat_dir / f"{key}.pt"
            torchaudio.save(str(out_wav), wav, args.sample_rate)
            torch.save(feat, out_feat)

            meta_item = {
                "filename": str(out_wav.relative_to(out_dir)),
                "feature_file": str(out_feat.relative_to(out_dir)),
                "sample_pack_key": pack,
                "instrument": inst_name,
                "type": type_name,
                "num_samples": int(wav.shape[-1]),
                "orig_relpath": str(rel),
            }
            if split_by_index is not None:
                meta_item["split"] = split_by_index[idx]
            meta[key] = meta_item

            kept += 1

        except Exception as e:
            failed += 1
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
