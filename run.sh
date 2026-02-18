#!/usr/bin/env bash
set -euo pipefail

WANDB_PROJECT="${WANDB_PROJECT:-drumblender}"
WANDB_NAME="${WANDB_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
WANDB_DIR="${WANDB_DIR:-/workspace/drumblender/logs/wandb}"
RUN_SEED="${RUN_SEED:-20260218}"

CFG="/workspace/drumblender/cfg/05_all_parallel.yaml"
DATA_DIR="/private/datasets/modal_features/processed_modal_flat"

# (선택) 파편화 완화. OOM 났던 환경이면 도움될 때가 있음
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# ### HIGHLIGHT: Use more validation batches for stable metrics and rotating val audio samples.
drumblender fit -c /workspace/drumblender/cfg/05_all_parallel.yaml \
  --seed_everything "$RUN_SEED" \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.precision 32 \
  --trainer.max_epochs -1 \
  --trainer.log_every_n_steps 40 \
  --trainer.num_sanity_val_steps 0 \
  --trainer.val_check_interval 0.1 \
  --trainer.limit_val_batches 8 \
  --trainer.default_root_dir /workspace/drumblender/lightning_logs \
  --trainer.logger pytorch_lightning.loggers.WandbLogger \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --trainer.logger.init_args.name "$WANDB_NAME" \
  --trainer.logger.init_args.save_dir "$WANDB_DIR" \
  --trainer.logger.init_args.log_model false \
  --data.class_path drumblender.data.AudioDataModule \
  --data.data_dir "$DATA_DIR" \
  --data.meta_file metadata.json \
  --data.dataset_class drumblender.data.AudioWithParametersDataset \
  --data.dataset_kwargs "{parameter_key: feature_file, split_strategy: sample_pack, expected_num_modes: 64, seed: $RUN_SEED}" \
  --data.seed "$RUN_SEED" \
  --data.sample_rate 48000 \
  --data.num_samples null \
  --data.batch_size 1 \
  --data.num_workers 6 \
  --data.skip_prepare_data true \
  --data.use_bucketing true \
  --data.bucket_boundaries "[48000, 96000, 192000, 384000]" \
  --data.drop_last true
