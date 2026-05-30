#!/usr/bin/env python3
"""
Pre-render transient-latent control sweeps for the static GitHub Pages demo.

This exporter uses the same descriptor-free controls as control.py and writes
docs/media/control/*.wav plus docs/data/control-data.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

DEFAULT_RUN_DIR = "../results/run_NOISEDAC_20260412_231956"
DEFAULT_SEED = 20260218
DEFAULT_SR = 48000

control = None


def _load_control_runtime() -> None:
    global control

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import control as control_module

    control_module._load_runtime()
    control = control_module


def _slugify(text: str) -> str:
    out: list[str] = []
    prev_dash = False
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif ch in {"-", "_"} or ch.isspace() or ch in {"/", "\\"}:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    return "".join(out).strip("-") or "sample"


def _write_wav(path: Path, audio, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    control.torchaudio.save(str(path), audio.detach().cpu(), int(sample_rate))


def _rel(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _select_indices(args: argparse.Namespace, dataset) -> list[int]:
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


def _pack_from_filename(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    return normalized.split("/", 1)[0] if "/" in normalized else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=Path("docs"))
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
    parser.add_argument("--clear", action="store_true", help="Remove previous docs/media/control before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_control_runtime()

    dev = control._device(args.device)
    model, cfg, ckpt, data_dir, audio_dir, run_dir = control._resolve_model_and_data(args)
    model.eval().to(dev)

    axis_dataset = control._make_dataset(data_dir, audio_dir, args, split=args.axis_split)
    test_dataset = control._make_dataset(data_dir, audio_dir, args, split=args.split)
    latent_values, latent_shape = control._collect_transient_latents(
        model=model,
        dataset=axis_dataset,
        device=dev,
        count=int(args.axis_samples),
        seed=int(args.seed),
    )
    axes = control._build_axes(
        model=model,
        axis_dataset=axis_dataset,
        device=dev,
        args=args,
        latent_values=latent_values,
        latent_shape=latent_shape,
    )
    if not axes:
        raise RuntimeError("No transient latent axes were built.")

    output_dir = args.output.resolve()
    media_root = output_dir / "media" / "control"
    data_root = output_dir / "data"
    if args.clear and media_root.exists():
        shutil.rmtree(media_root)
    media_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    indices = _select_indices(args, test_dataset)
    states = control._build_sample_states(model, test_dataset, indices, dev)
    value_grid = control._control_values()
    demos: list[dict[str, Any]] = []

    for sample_order, sample in enumerate(states, start=1):
        filename = sample.sample_filename
        pack = _pack_from_filename(filename)
        digest = hashlib.sha1(f"{sample.sample_index}:{filename}".encode("utf-8")).hexdigest()[:8]
        sample_slug = f"{sample_order:02d}-{_slugify(filename)[:54]}-{digest}"
        sample_dir = media_root / sample_slug

        target_path = sample_dir / "target.wav"
        baseline_path = sample_dir / "reconstruction-neutral.wav"
        _write_wav(target_path, sample.target_waveform, args.sample_rate)
        _write_wav(baseline_path, control._render_neutral(model, sample), args.sample_rate)

        for axis in axes:
            axis_slug = _slugify(axis.name)
            axis_dir = sample_dir / axis_slug
            variants = []
            for label, signed_value in value_grid:
                if abs(float(signed_value)) <= 1e-12:
                    audio = control._render_neutral(model, sample)
                else:
                    audio = control._render_with_axis(
                        model=model,
                        sample=sample,
                        axis=axis,
                        signed_value=float(signed_value) * float(args.control_strength),
                    )
                value_id = label.replace("+", "p").replace("-", "m").replace(".", "_")
                variant_path = axis_dir / f"value-{value_id}.wav"
                _write_wav(variant_path, audio, args.sample_rate)
                variants.append(
                    {
                        "sigma": float(signed_value),
                        "label": label,
                        "audio": _rel(variant_path, output_dir),
                    }
                )

            demos.append(
                {
                    "id": f"{sample_slug}-{axis_slug}",
                    "title": f"{axis.name} | {pack or 'sample'} #{sample_order}",
                    "module": axis.method,
                    "axis": axis.name,
                    "axis_note": axis.note,
                    "sample_index": int(sample.sample_index),
                    "source_filename": filename,
                    "pack": pack,
                    "target": _rel(target_path, output_dir),
                    "baseline": _rel(baseline_path, output_dir),
                    "variants": variants,
                }
            )

    payload = {
        "site_title": "Transient Latent Control Demo",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_from": str(run_dir),
        "methods": args.methods,
        "value_semantics": "signed transient-latent displacement; 0 is neutral, +/-1 is one latent projection std",
        "control_strength": float(args.control_strength),
        "axis_group_size": int(args.axis_group_size),
        "sample_rate": int(args.sample_rate),
        "checkpoint": str(ckpt),
        "config": str(cfg),
        "data_dir": str(data_dir),
        "audio_dir": str(audio_dir),
        "latent_shape": list(latent_shape),
        "axis_count": len(axes),
        "demo_count": len(demos),
        "demos": demos,
    }
    out_json = data_root / "control-data.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[control-demo] wrote {len(demos)} demos -> {out_json}")


if __name__ == "__main__":
    main()
