#!/usr/bin/env bash
set -euo pipefail

cd /workspace/drumblender

CFG="${CFG:-/workspace/drumblender/cfg/05_all_parallel.yaml}"
DATA_DIR="${DATA_DIR:-/mnt/datasets/modal_features/processed_modal_flat}"
SEED="${SEED:-20260218}"
BATCH_SIZE="${BATCH_SIZE:-2}"
USE_BUCKETING="${USE_BUCKETING:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ ! -f "$CFG" ]]; then
  echo "Config not found: $CFG"
  exit 1
fi

if [[ ! -f "$DATA_DIR/metadata.json" ]]; then
  echo "Dataset metadata not found: $DATA_DIR/metadata.json"
  exit 1
fi

echo "[debug_fit_2gpu_nolog] starting..."
drumblender fit -c "$CFG" \
  --seed_everything "$SEED" \
  --trainer.accelerator gpu \
  --trainer.devices 2 \
  --trainer.strategy ddp \
  --trainer.replace_sampler_ddp false \
  --trainer.precision 32 \
  --trainer.max_epochs 1 \
  --trainer.limit_train_batches 10 \
  --trainer.limit_val_batches 2 \
  --trainer.val_check_interval 1.0 \
  --trainer.num_sanity_val_steps 0 \
  --trainer.enable_checkpointing false \
  --trainer.logger false \
  --trainer.callbacks null \
  --trainer.log_every_n_steps 5 \
  --data.class_path drumblender.data.AudioDataModule \
  --data.data_dir "$DATA_DIR" \
  --data.meta_file metadata.json \
  --data.dataset_class drumblender.data.AudioWithParametersDataset \
  --data.dataset_kwargs "{parameter_key: feature_file, split_strategy: sample_pack, expected_num_modes: 64, seed: $SEED}" \
  --data.seed "$SEED" \
  --data.sample_rate 48000 \
  --data.num_samples null \
  --data.batch_size "$BATCH_SIZE" \
  --data.num_workers "$NUM_WORKERS" \
  --data.skip_prepare_data true \
  --data.use_bucketing "$USE_BUCKETING" \
  --data.drop_last true
echo "[debug_fit_2gpu_nolog] done."
