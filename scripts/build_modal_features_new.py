from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
import threading
from typing import Dict, List, Tuple

import torch
import torchaudio
from tqdm import tqdm

from drumblender.utils.modal_analysis_new import CQTModalAnalysis


_WORKER_CFG: Dict | None = None
_WORKER_MODAL: CQTModalAnalysis | None = None
_WORKER_PROGRESS_QUEUE = None
_THREAD_LOCAL = threading.local()


def stable_id(rel_path: str) -> str:
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
    rel = wav_path.relative_to(processed_root)
    parts = rel.parts

    type_name = "custom"
    inst_name = "unlabeled"

    inner_dirs = list(parts[:-1])
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
    if train < 0.0 or val < 0.0 or (train + val) > 1.0:
        raise ValueError(
            "Invalid split ratios: require train >= 0, val >= 0, train+val <= 1"
        )

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


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count - 1 if cpu_count > 1 else 1))


def default_chunk_size() -> int:
    return 32


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Build modal features with the new HPSS + Kalman + confidence tracker. "
            "Optimized for faster preprocessing via optional audio-copy skipping "
            "and multiprocessing."
        )
    )

    ap.add_argument(
        "--processed_root",
        type=str,
        default="../datasets/processed",
        help="Source preprocessed audio root.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="../datasets_new/modal_features/processed_modal_flat",
        help="Output directory for flat modal-feature dataset.",
    )
    ap.add_argument("--meta_name", type=str, default="metadata.json")

    ap.add_argument("--sample_rate", type=int, default=48000)
    ap.add_argument("--num_modes", type=int, default=64)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--fmin", type=int, default=20)
    ap.add_argument("--n_bins", type=int, default=240)
    ap.add_argument("--bins_per_octave", type=int, default=24)
    ap.add_argument("--min_length", type=int, default=10)
    ap.add_argument("--threshold_db", type=float, default=-80.0)

    ap.add_argument("--mask_threshold", type=float, default=0.8)
    ap.add_argument("--kalman_k", type=float, default=0.4)
    ap.add_argument("--kalman_kv", type=float, default=0.4)
    ap.add_argument("--confidence_threshold", type=float, default=0.5)
    ap.add_argument("--h_kernel", type=int, default=31)
    ap.add_argument("--p_kernel", type=int, default=31)

    ap.add_argument("--seed", type=int, default=5152845)
    ap.add_argument("--max_files", type=int, default=0, help="0 = all files")
    ap.add_argument(
        "--pack_depth",
        type=int,
        default=1,
        help="Number of top-level path segments under processed_root used as pack id.",
    )
    ap.add_argument(
        "--write_split",
        action="store_true",
        help="Write split labels into metadata during preprocessing.",
    )

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
    ap.add_argument(
        "--pad_to",
        type=int,
        default=0,
        help="If >0, right-pad audio shorter than this to this length (samples).",
    )
    ap.add_argument(
        "--min_duration_ms",
        type=float,
        default=0.0,
        help="If >0, skip files shorter than this (ms). Set 0 to not skip.",
    )

    ap.add_argument(
        "--num_workers",
        type=int,
        default=default_num_workers(),
        help="Number of worker processes for parallel modal extraction.",
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=default_chunk_size(),
        help="How many files each parallel task should process per chunk.",
    )
    ap.add_argument(
        "--copy_audio",
        dest="copy_audio",
        action="store_true",
        help="Copy audio into out_dir/audio for a fully standalone flat dataset.",
    )
    ap.add_argument(
        "--no_copy_audio",
        dest="copy_audio",
        action="store_false",
        help="Do not copy audio; metadata points back to processed_root audio paths.",
    )
    ap.set_defaults(copy_audio=False)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute feature files even if they already exist.",
    )
    ap.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="Write metadata checkpoint every N newly completed items.",
    )

    return ap


def make_worker_config(args: argparse.Namespace, processed_root: Path, out_dir: Path) -> Dict:
    return {
        "processed_root": str(processed_root),
        "out_dir": str(out_dir),
        "sample_rate": int(args.sample_rate),
        "num_modes": int(args.num_modes),
        "hop_length": int(args.hop_length),
        "fmin": float(args.fmin),
        "n_bins": int(args.n_bins),
        "bins_per_octave": int(args.bins_per_octave),
        "min_length": int(args.min_length),
        "threshold_db": float(args.threshold_db),
        "mask_threshold": float(args.mask_threshold),
        "kalman_k": float(args.kalman_k),
        "kalman_kv": float(args.kalman_kv),
        "confidence_threshold": float(args.confidence_threshold),
        "h_kernel": int(args.h_kernel),
        "p_kernel": int(args.p_kernel),
        "pad_short": bool(args.pad_short),
        "pad_to": int(args.pad_to),
        "min_duration_ms": float(args.min_duration_ms),
        "copy_audio": bool(args.copy_audio),
        "overwrite": bool(args.overwrite),
    }


def init_worker(cfg: Dict, progress_queue=None) -> None:
    global _WORKER_CFG, _WORKER_MODAL, _WORKER_PROGRESS_QUEUE
    _WORKER_CFG = cfg
    _WORKER_MODAL = None
    _WORKER_PROGRESS_QUEUE = progress_queue

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def build_modal_from_cfg(cfg: Dict) -> CQTModalAnalysis:
    return CQTModalAnalysis(
        cfg["sample_rate"],
        hop_length=cfg["hop_length"],
        fmin=cfg["fmin"],
        n_bins=cfg["n_bins"],
        bins_per_octave=cfg["bins_per_octave"],
        min_length=cfg["min_length"],
        num_modes=cfg["num_modes"],
        threshold=cfg["threshold_db"],
        p_kernel=cfg["p_kernel"],
    )


def get_worker_modal() -> CQTModalAnalysis:
    global _WORKER_MODAL
    assert _WORKER_CFG is not None
    if _WORKER_MODAL is None:
        _WORKER_MODAL = build_modal_from_cfg(_WORKER_CFG)
    return _WORKER_MODAL


def get_thread_modal(cfg: Dict) -> CQTModalAnalysis:
    modal = getattr(_THREAD_LOCAL, "modal", None)
    if modal is None:
        modal = build_modal_from_cfg(cfg)
        _THREAD_LOCAL.modal = modal
    return modal


def right_pad_to(w: torch.Tensor, target: int) -> torch.Tensor:
    length = int(w.shape[-1])
    if length >= target:
        return w
    return torch.nn.functional.pad(w, (0, target - length))


def num_frames_from_len(length: int, hop: int) -> int:
    return max(1, math.ceil(length / hop))


def infer_required_length_from_padding_error(msg: str, current_len: int) -> int:
    match = re.search(r"padding\s+\((\d+),\s*(\d+)\)", msg)
    if match:
        pad_l = int(match.group(1))
        pad_r = int(match.group(2))
        return max(current_len, pad_l + 1, pad_r + 1)
    return max(current_len + 1, current_len * 2)


def build_audio_reference(wav_path: Path, out_dir: Path) -> str:
    try:
        return os.path.relpath(str(wav_path), str(out_dir))
    except ValueError:
        return str(wav_path)


def build_feature_relpath(key: str) -> Path:
    return Path("features") / f"{key}.pt"


def build_audio_relpath(key: str) -> Path:
    return Path("audio") / f"{key}.wav"


def extract_feat_from_waveform(w: torch.Tensor, modal: CQTModalAnalysis, cfg: Dict) -> torch.Tensor:
    modal_freqs, modal_amps, modal_phases = modal(w)

    if modal_freqs.numel() == 0 or modal_freqs.shape[1] == 0:
        num_frames = num_frames_from_len(int(w.shape[-1]), cfg["hop_length"])
        return w.new_zeros((3, 0, num_frames))

    modal_freqs = 2 * torch.pi * modal_freqs / cfg["sample_rate"]
    feat = torch.stack([modal_freqs, modal_amps, modal_phases])
    feat = feat.squeeze(1)
    return feat


def build_metadata_item(job: Dict, cfg: Dict, rel: Path, key: str, num_samples: int) -> Dict:
    out_dir = Path(cfg["out_dir"])
    wav_path = Path(job["wav_path"])
    feat_rel = build_feature_relpath(key)
    if cfg["copy_audio"]:
        filename = str(build_audio_relpath(key))
    else:
        filename = build_audio_reference(wav_path, out_dir)

    meta_item = {
        "filename": filename,
        "feature_file": str(feat_rel),
        "sample_pack_key": job["pack"],
        "instrument": job["inst_name"],
        "type": job["type_name"],
        "num_samples": int(num_samples),
        "orig_relpath": str(rel),
    }
    if job["split"] is not None:
        meta_item["split"] = job["split"]
    return meta_item


def chunk_jobs(jobs: List[Dict], chunk_size: int) -> List[List[Dict]]:
    chunk_size = max(1, int(chunk_size))
    return [jobs[i : i + chunk_size] for i in range(0, len(jobs), chunk_size)]


def load_existing_metadata(meta_path: Path) -> Dict[str, Dict]:
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(meta_path: Path, meta: Dict[str, Dict]) -> None:
    ordered_meta = {key: meta[key] for key in sorted(meta.keys())}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(ordered_meta, f, ensure_ascii=False)


def notify_worker_progress() -> None:
    if _WORKER_PROGRESS_QUEUE is None:
        return
    try:
        _WORKER_PROGRESS_QUEUE.put(1)
    except Exception:
        pass


def consume_progress(progress_queue, pbar, expected: int, stop_event: threading.Event) -> None:
    completed = 0
    while completed < expected:
        try:
            item = progress_queue.get(timeout=0.5)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue
        except Exception:
            if stop_event.is_set():
                break
            continue

        if item is None:
            if stop_event.is_set():
                break
            continue

        completed += 1
        pbar.update(1)


def process_one_common(job: Dict, cfg: Dict, modal: CQTModalAnalysis) -> Dict:
    processed_root = Path(cfg["processed_root"])
    out_dir = Path(cfg["out_dir"])
    wav_path = Path(job["wav_path"])
    rel = wav_path.relative_to(processed_root)
    key = job["key"]

    try:
        feat_rel = build_feature_relpath(key)
        feat_path = out_dir / feat_rel
        audio_rel = build_audio_relpath(key)
        audio_path = out_dir / audio_rel

        if (not cfg["overwrite"]) and feat_path.exists():
            num_samples_hint = job.get("num_samples_hint")
            if num_samples_hint is None:
                num_samples_hint = int(torchaudio.info(str(wav_path)).num_frames)
            meta_item = build_metadata_item(job, cfg, rel, key, num_samples_hint)
            if cfg["copy_audio"] and (not audio_path.exists()):
                wav, sr = torchaudio.load(str(wav_path))
                if sr != cfg["sample_rate"]:
                    raise ValueError(
                        f"sample_rate mismatch: {sr} != {cfg['sample_rate']} for {rel}"
                    )
                wav = wav[:1, :]
                torchaudio.save(str(audio_path), wav, cfg["sample_rate"])
            return {"ok": True, "key": key, "meta_item": meta_item, "skipped": True}

        wav, sr = torchaudio.load(str(wav_path))
        if wav.ndim != 2:
            raise ValueError(f"bad wav shape: {tuple(wav.shape)}")
        if sr != cfg["sample_rate"]:
            raise ValueError(
                f"sample_rate mismatch: {sr} != {cfg['sample_rate']} for {rel}"
            )
        if wav.shape[0] == 0:
            raise ValueError("empty channel dim")
        wav = wav[:1, :]

        if cfg["min_duration_ms"] > 0:
            min_samples = int(cfg["sample_rate"] * (cfg["min_duration_ms"] / 1000.0))
            if wav.shape[-1] < min_samples:
                raise ValueError(f"too_short: {wav.shape[-1]} < {min_samples} samples")

        if cfg["pad_to"] > 0:
            wav = right_pad_to(wav, cfg["pad_to"])

        try:
            feat = extract_feat_from_waveform(wav, modal, cfg)
        except RuntimeError as e:
            msg = str(e)
            if (not cfg["pad_short"]) or (
                "Padding size should be less than the corresponding input dimension"
                not in msg
            ):
                raise
            target = infer_required_length_from_padding_error(msg, int(wav.shape[-1]))
            feat = extract_feat_from_waveform(right_pad_to(wav, target), modal, cfg)

        p_dim, m_dim, f_dim = feat.shape
        if m_dim < cfg["num_modes"]:
            pad = feat.new_zeros((p_dim, cfg["num_modes"] - m_dim, f_dim))
            feat = torch.cat([feat, pad], dim=1)
        elif m_dim > cfg["num_modes"]:
            feat = feat[:, : cfg["num_modes"], :]

        torch.save(feat, feat_path)

        if cfg["copy_audio"]:
            torchaudio.save(str(audio_path), wav, cfg["sample_rate"])

        meta_item = build_metadata_item(job, cfg, rel, key, int(wav.shape[-1]))
        return {"ok": True, "key": key, "meta_item": meta_item, "skipped": False}
    except Exception as e:
        return {"ok": False, "rel": str(rel), "error": repr(e)}


def process_one(job: Dict) -> Dict:
    assert _WORKER_CFG is not None
    cfg = _WORKER_CFG
    modal = get_worker_modal()
    return process_one_common(job, cfg, modal)


def process_one_threaded(job: Dict, cfg: Dict) -> Dict:
    modal = get_thread_modal(cfg)
    return process_one_common(job, cfg, modal)


def process_chunk(jobs: List[Dict]) -> List[Dict]:
    results = []
    for job in jobs:
        result = process_one(job)
        results.append(result)
        notify_worker_progress()
    return results


def process_chunk_threaded(jobs: List[Dict], cfg: Dict, progress_queue=None) -> List[Dict]:
    modal = get_thread_modal(cfg)
    results = []
    for job in jobs:
        result = process_one_common(job, cfg, modal)
        results.append(result)
        if progress_queue is not None:
            try:
                progress_queue.put(1)
            except Exception:
                pass
    return results


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()

    processed_root = Path(args.processed_root)
    if not processed_root.exists():
        raise FileNotFoundError(processed_root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_audio:
        (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / args.meta_name

    wavs = list_wavs(processed_root)
    if args.max_files and args.max_files > 0:
        wavs = wavs[: args.max_files]

    print(f"[scan] {processed_root} -> {len(wavs)} wavs")
    print(f"[out]  {out_dir}")
    print(
        f"[mode] copy_audio={args.copy_audio} num_workers={args.num_workers} "
        f"chunk_size={args.chunk_size} overwrite={args.overwrite}"
    )
    if len(wavs) == 0:
        raise RuntimeError("No wavs found.")

    existing_meta = load_existing_metadata(meta_path) if (not args.overwrite) else {}
    typed: List[Tuple[str, str, str]] = []
    packs: List[str] = []
    for wav_path in wavs:
        type_name, inst_name, pack = make_pack_key(
            processed_root,
            wav_path,
            pack_depth=args.pack_depth,
        )
        typed.append((type_name, inst_name, pack))
        packs.append(pack)

    split_by_index = None
    if args.write_split:
        split_by_index = make_splits_within_pack(packs, seed=args.seed)

    jobs = []
    meta: Dict[str, Dict] = dict(existing_meta)
    skipped_existing = 0
    for idx, (wav_path, (type_name, inst_name, pack)) in enumerate(zip(wavs, typed)):
        rel = wav_path.relative_to(processed_root)
        key = stable_id(str(rel))
        job = {
            "key": key,
            "wav_path": str(wav_path),
            "type_name": type_name,
            "inst_name": inst_name,
            "pack": pack,
            "split": None if split_by_index is None else split_by_index[idx],
            "num_samples_hint": None,
        }
        feat_path = out_dir / build_feature_relpath(key)
        audio_path = out_dir / build_audio_relpath(key)
        if (
            (not args.overwrite)
            and key in existing_meta
            and feat_path.exists()
            and ((not args.copy_audio) or audio_path.exists())
        ):
            meta[key] = existing_meta[key]
            skipped_existing += 1
            continue
        jobs.append(job)

    worker_cfg = make_worker_config(args, processed_root, out_dir)
    failed = 0
    newly_completed = 0

    def handle_result(result: Dict) -> None:
        nonlocal failed, newly_completed
        if result["ok"]:
            meta[result["key"]] = result["meta_item"]
            if not result.get("skipped", False):
                newly_completed += 1
                if args.save_every > 0 and newly_completed % args.save_every == 0:
                    save_metadata(meta_path, meta)
        else:
            failed += 1
            print("fail:", result["rel"], "->", result["error"])

    print(f"[resume] reused={skipped_existing} pending={len(jobs)}")
    if len(jobs) == 0:
        save_metadata(meta_path, meta)
        print("[done] kept:", len(meta), "failed:", failed)
        print("out_dir:", out_dir)
        print("meta:", meta_path)
        return

    job_chunks = chunk_jobs(jobs, args.chunk_size)
    progress_bar = tqdm(
        total=len(wavs),
        initial=skipped_existing,
        desc="modal-new",
        unit="file",
        dynamic_ncols=True,
    )

    if args.num_workers <= 1:
        init_worker(worker_cfg)
        for chunk in job_chunks:
            for result in process_chunk(chunk):
                handle_result(result)
                progress_bar.update(1)
            progress_bar.set_postfix(kept=len(meta), failed=failed, done=newly_completed)
    else:
        used_thread_fallback = False
        progress_thread = None
        stop_event = threading.Event()
        progress_queue = None
        try:
            ctx = mp.get_context("spawn")
            progress_queue = ctx.Queue()
            progress_thread = threading.Thread(
                target=consume_progress,
                args=(progress_queue, progress_bar, len(jobs), stop_event),
                daemon=True,
            )
            progress_thread.start()
            with ProcessPoolExecutor(
                max_workers=args.num_workers,
                mp_context=ctx,
                initializer=init_worker,
                initargs=(worker_cfg, progress_queue),
            ) as pool:
                futures = [pool.submit(process_chunk, chunk) for chunk in job_chunks]
                for future in as_completed(futures):
                    for result in future.result():
                        handle_result(result)
                    progress_bar.set_postfix(
                        kept=len(meta), failed=failed, done=newly_completed
                    )
        except (PermissionError, OSError) as e:
            used_thread_fallback = True
            if progress_thread is not None:
                stop_event.set()
                try:
                    progress_queue.put(None)
                except Exception:
                    pass
                progress_thread.join(timeout=2.0)
            print(
                "[warn] ProcessPoolExecutor unavailable in this environment; "
                f"falling back to threads. reason={repr(e)}"
            )
        else:
            stop_event.set()
            try:
                progress_queue.put(None)
            except Exception:
                pass
            if progress_thread is not None:
                progress_thread.join(timeout=5.0)

        if used_thread_fallback:
            progress_queue = queue.Queue()
            stop_event = threading.Event()
            progress_thread = threading.Thread(
                target=consume_progress,
                args=(progress_queue, progress_bar, len(jobs), stop_event),
                daemon=True,
            )
            progress_thread.start()
            with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
                futures = [
                    pool.submit(process_chunk_threaded, chunk, worker_cfg, progress_queue)
                    for chunk in job_chunks
                ]
                for future in as_completed(futures):
                    for result in future.result():
                        handle_result(result)
                    progress_bar.set_postfix(
                        kept=len(meta), failed=failed, done=newly_completed
                    )
            stop_event.set()
            try:
                progress_queue.put(None)
            except Exception:
                pass
            progress_thread.join(timeout=5.0)

    save_metadata(meta_path, meta)
    progress_bar.close()
    print("[done] kept:", len(meta), "failed:", failed)
    print("out_dir:", out_dir)
    print("meta:", meta_path)


if __name__ == "__main__":
    mp.freeze_support()
    main()
