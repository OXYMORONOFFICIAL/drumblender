#!/usr/bin/env python3
"""
Export and package reconstruction artifacts for easy sharing.

Outputs include:
- Reconstruction wav files (and optional targets)
- Selected checkpoint
- Evaluation summary/loss stats on the selected split
- Per-file loss CSV
- Config files and useful training logs
- A final tar.gz archive for transfer (scp-friendly)
"""

import argparse
import csv
import importlib
import json
import shutil
import statistics
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchaudio
import yaml
from torchmetrics import Metric
from tqdm import tqdm

from drumblender.data.audio import AudioWithParametersDataset
from drumblender.utils.model import load_model


def _optional_int(value: str) -> Optional[int]:
    v = value.strip().lower()
    if v in {"none", "null", ""}:
        return None
    return int(value)


def _resolve_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _instantiate_from_spec(spec: Any):
    if isinstance(spec, dict):
        if "class_path" in spec:
            cls = _resolve_class(spec["class_path"])
            init_args = spec.get("init_args", {})
            resolved = {k: _instantiate_from_spec(v) for k, v in init_args.items()}
            return cls(**resolved)
        return {k: _instantiate_from_spec(v) for k, v in spec.items()}
    if isinstance(spec, list):
        return [_instantiate_from_spec(x) for x in spec]
    return spec


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_training_context(ckpt_path: Path, package_root: Path) -> None:
    """
    Copy nearby training context files from the run directory.
    """
    # ### HIGHLIGHT: Infer run directory from ".../checkpoints/<file>.ckpt".
    run_dir: Optional[Path] = None
    if ckpt_path.parent.name == "checkpoints":
        run_dir = ckpt_path.parent.parent

    ctx_dir = package_root / "training_context"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    # Repository-level scripts/configs often needed to reproduce the run.
    for repo_file in [
        Path("run.sh"),
        Path("run_vessl.sh"),
        Path("cfg/05_all_parallel.yaml"),
        Path("cfg/metrics/drumblender_metrics.yaml"),
    ]:
        if repo_file.exists():
            _copy_if_exists(repo_file, ctx_dir / "repo" / repo_file)

    # LightningCLI transient config (if present in current working directory).
    _copy_if_exists(Path("config.yaml"), ctx_dir / "repo" / "config.yaml")

    if run_dir is None or not run_dir.exists():
        return

    # Copy useful run-local logs without pulling full checkpoints directory.
    wanted_patterns = [
        "metrics.csv",
        "hparams.yaml",
        "model-config.yaml",
        "config.yaml",
        "*.log",
        "*.json",
        "*.jsonl",
        "events.out.tfevents*",
    ]
    for pattern in wanted_patterns:
        for src in run_dir.rglob(pattern):
            if not src.is_file():
                continue
            # skip checkpoint files; selected ckpt is copied separately
            if src.suffix == ".ckpt":
                continue
            rel = src.relative_to(run_dir)
            _copy_if_exists(src, ctx_dir / "run_dir" / rel)


def _collect_git_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True
        ).strip()
        info["commit"] = head
    except Exception:
        info["commit"] = None

    try:
        status = subprocess.check_output(
            ["git", "status", "--short"], stderr=subprocess.STDOUT, text=True
        )
        info["status_short"] = status.splitlines()
    except Exception:
        info["status_short"] = []

    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root")
    parser.add_argument("--meta-file", type=str, default="metadata.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--split-strategy", type=str, default="sample_pack", choices=["sample_pack", "random"]
    )
    parser.add_argument("--parameter-key", type=str, default="feature_file")
    parser.add_argument("--expected-num-modes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument(
        "--num-samples",
        type=_optional_int,
        default=None,
        help="Fixed-length mode if set, else variable-length (use none/null)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="recon_bundle",
        help="Directory to save packaged artifacts",
    )
    parser.add_argument(
        "--save-target",
        action="store_true",
        help="Also save ground-truth target wav files",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap for quick checks",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--metrics-config",
        type=str,
        default="cfg/metrics/drumblender_metrics.yaml",
        help="Metric config YAML used for evaluation summary",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation summary computation",
    )
    parser.add_argument(
        "--make-tar",
        action="store_true",
        help="Create <output-dir>.tar.gz after export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    recon_root = output_dir / "recon"
    target_root = output_dir / "target"
    eval_root = output_dir / "evaluation"
    ckpt_root = output_dir / "checkpoints"
    config_root = output_dir / "configs"
    recon_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    if args.save_target:
        target_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.config, args.ckpt, include_data=False)
    model = model.to(device)
    model.eval()

    # ### HIGHLIGHT: Keep a copy of configs used for this export.
    _copy_if_exists(Path(args.config), config_root / Path(args.config).name)
    _copy_if_exists(Path(args.metrics_config), config_root / Path(args.metrics_config).name)

    ckpt_path = Path(args.ckpt)
    _copy_if_exists(ckpt_path, ckpt_root / ckpt_path.name)
    _copy_training_context(ckpt_path, output_dir)

    dataset = AudioWithParametersDataset(
        data_dir=args.data_dir,
        meta_file=args.meta_file,
        sample_rate=args.sample_rate,
        num_samples=args.num_samples,
        split=args.split,
        split_strategy=args.split_strategy,
        parameter_key=args.parameter_key,
        expected_num_modes=args.expected_num_modes,
        seed=args.seed,
    )

    # Instantiate evaluation metrics from YAML.
    metric_modules: Dict[str, Any] = {}
    if not args.no_eval:
        with open(args.metrics_config, "r", encoding="utf-8") as f:
            metrics_spec = yaml.safe_load(f)
        metric_container = _instantiate_from_spec(metrics_spec)
        if hasattr(metric_container, "items"):
            metric_modules = dict(metric_container.items())
        else:
            raise RuntimeError("metrics-config must instantiate a module dict-like container")

        for m in metric_modules.values():
            if hasattr(m, "to"):
                m.to(device)

    limit = len(dataset) if args.max_items is None else min(len(dataset), args.max_items)
    manifest_path = output_dir / "manifest.csv"
    per_file_loss_path = eval_root / "per_file_loss.csv"

    losses = []
    raw_metric_sums: Dict[str, float] = {}
    raw_metric_counts: Dict[str, int] = {}

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file, per_file_loss_path.open(
        "w", newline="", encoding="utf-8"
    ) as loss_file:
        manifest_writer = csv.writer(manifest_file)
        loss_writer = csv.writer(loss_file)
        manifest_writer.writerow(
            [
                "index",
                "meta_key",
                "source_filename",
                "length",
                "recon_path",
                "target_path",
            ]
        )
        loss_writer.writerow(["index", "meta_key", "source_filename", "length", "test/loss"])

        for idx in tqdm(range(limit), desc="export"):
            meta_key = dataset.file_list[idx]
            meta = dataset.metadata[meta_key]
            src_rel = Path(meta["filename"])

            waveform, params, length = dataset[idx]
            length_i = int(length)

            with torch.no_grad():
                x = waveform.unsqueeze(0).to(device)
                p = params.unsqueeze(0).to(device)
                lengths = torch.tensor([length_i], dtype=torch.long, device=device)
                y_hat = model(x, p, lengths=lengths)
                loss_value = float(model.loss_fn(y_hat, x).detach().cpu())

            losses.append(loss_value)

            # Evaluate additional metrics.
            if not args.no_eval:
                for name, metric in metric_modules.items():
                    if isinstance(metric, Metric):
                        metric.update(y_hat, x)
                    else:
                        value = float(metric(y_hat, x).detach().cpu())
                        raw_metric_sums[name] = raw_metric_sums.get(name, 0.0) + value
                        raw_metric_counts[name] = raw_metric_counts.get(name, 0) + 1

            recon = y_hat.squeeze(0).detach().cpu()[:, :length_i]
            target = waveform[:, :length_i]

            recon_path = recon_root / src_rel
            recon_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(recon_path), recon, args.sample_rate)

            target_path_str = ""
            if args.save_target:
                target_path = target_root / src_rel
                target_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(target_path), target, args.sample_rate)
                target_path_str = str(target_path)

            manifest_writer.writerow(
                [
                    idx,
                    meta_key,
                    str(src_rel),
                    length_i,
                    str(recon_path),
                    target_path_str,
                ]
            )
            loss_writer.writerow([idx, meta_key, str(src_rel), length_i, loss_value])

    # Build summary metrics payload.
    summary = {
        "export_time_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "metrics_config": args.metrics_config,
        "ckpt": args.ckpt,
        "data_dir": args.data_dir,
        "meta_file": args.meta_file,
        "split": args.split,
        "split_strategy": args.split_strategy,
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "num_samples": args.num_samples,
        "num_items": limit,
        "git": _collect_git_info(),
        "metrics": {},
    }

    if len(losses) > 0:
        summary["metrics"]["test/loss"] = float(sum(losses) / len(losses))
        summary["loss_stats"] = {
            "mean": float(statistics.fmean(losses)),
            "min": float(min(losses)),
            "max": float(max(losses)),
            "median": float(statistics.median(losses)),
            "p95": float(sorted(losses)[max(0, int(0.95 * len(losses)) - 1)]),
            "count": len(losses),
        }

    if not args.no_eval:
        for name, metric in metric_modules.items():
            if isinstance(metric, Metric):
                summary["metrics"][f"test/{name}"] = float(metric.compute().detach().cpu())
            else:
                c = raw_metric_counts.get(name, 0)
                if c > 0:
                    summary["metrics"][f"test/{name}"] = float(raw_metric_sums[name] / c)

    summary_path = eval_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also write a CSV one-liner for easy spreadsheet loading.
    summary_csv_path = eval_root / "summary_metrics.csv"
    metric_keys = sorted(summary["metrics"].keys())
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(metric_keys)
        w.writerow([summary["metrics"][k] for k in metric_keys])

    if args.make_tar:
        tar_path = output_dir.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)
        print(f"[OK] archive: {tar_path}")

    print(f"[OK] exported {limit} items to {output_dir}")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] eval summary: {summary_path}")


if __name__ == "__main__":
    main()
