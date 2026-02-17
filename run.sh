#!/usr/bin/env bash

# 터미널 꺼지는 거 방지용: set -euo pipefail 제거(원하면 다시 넣어도 됨)
cd /workspace/drumblender || exit 1
mkdir -p logs

export WANDB_PROJECT="drumblender"
export WANDB_NAME="all_parallel_varlen_48k_bs1_bucket_2080ti"
export WANDB_DIR="/workspace/drumblender/logs"

# (선택) 파편화 완화. OOM 났던 환경이면 도움될 때가 있음
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# ### HIGHLIGHT: Use more validation batches for stable metrics and rotating val audio samples.
drumblender fit -c /workspace/drumblender/cfg/05_all_parallel.yaml \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.precision 32 \
  --trainer.max_epochs -1 \
  --trainer.log_every_n_steps 10 \
  --trainer.num_sanity_val_steps 0 \
  --trainer.val_check_interval 0.25 \
  --trainer.limit_val_batches 8 \
  --trainer.default_root_dir /workspace/drumblender/lightning_logs \
  --trainer.logger pytorch_lightning.loggers.WandbLogger \
  --trainer.logger.init_args.project "$WANDB_PROJECT" \
  --trainer.logger.init_args.name "$WANDB_NAME" \
  --trainer.logger.init_args.save_dir "$WANDB_DIR" \
  --trainer.logger.init_args.log_model false \
  --data.class_path drumblender.data.AudioDataModule \
  --data.data_dir /private/datasets/modal_features/processed_modal_flat \
  --data.meta_file metadata.json \
  --data.dataset_class drumblender.data.AudioWithParametersDataset \
  --data.dataset_kwargs "{parameter_key: feature_file, split_strategy: sample_pack, expected_num_modes: 64}" \
  --data.sample_rate 48000 \
  --data.num_samples null \
  --data.batch_size 1 \
  --data.num_workers 6 \
  --data.skip_prepare_data true \
  --data.use_bucketing true \
  --data.bucket_boundaries "[48000, 96000, 192000, 384000]" \
  --data.drop_last true
