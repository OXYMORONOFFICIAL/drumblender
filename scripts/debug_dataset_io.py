#!/usr/bin/env python3
"""
Minimal dataset I/O smoke test for drumblender.

This script does not launch training. It only verifies:
1) metadata loading
2) waveform/feature loading
3) collate behavior for one batch
"""

import argparse
import random
import time

import torch
from torch.utils.data import DataLoader

from drumblender.data.audio import AudioWithParametersDataset
from drumblender.data.collate import pad_audio_params_collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/datasets/modal_features/processed_modal_flat",
    )
    parser.add_argument("--meta-file", type=str, default="metadata.json")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-items", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260218)
    args = parser.parse_args()

    t0 = time.time()
    ds = AudioWithParametersDataset(
        data_dir=args.data_dir,
        meta_file=args.meta_file,
        sample_rate=args.sample_rate,
        num_samples=None,
        split="train",
        split_strategy="sample_pack",
        parameter_key="feature_file",
        expected_num_modes=64,
        seed=args.seed,
    )
    t1 = time.time()
    print(f"[OK] dataset init: len={len(ds)} in {t1 - t0:.2f}s")

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: max(1, min(args.n_items, len(ds)))]

    total_item_time = 0.0
    for i, idx in enumerate(indices):
        s0 = time.time()
        w, p, L = ds[idx]
        s1 = time.time()
        total_item_time += s1 - s0
        print(
            f"[ITEM {i:02d}] idx={idx} wav={tuple(w.shape)} params={tuple(p.shape)} "
            f"len={int(L)} t={s1 - s0:.3f}s"
        )

    print(
        f"[OK] item load avg: {total_item_time / len(indices):.3f}s over {len(indices)} items"
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_audio_params_collate,
        drop_last=True,
        pin_memory=True,
    )
    b0 = time.time()
    batch = next(iter(dl))
    b1 = time.time()
    w, p, L = batch
    print(
        f"[OK] dataloader 1st batch in {b1 - b0:.3f}s "
        f"wav={tuple(w.shape)} params={tuple(p.shape)} lengths={tuple(L.shape)}"
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

