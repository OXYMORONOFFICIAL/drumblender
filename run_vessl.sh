#!/usr/bin/env bash
set -euo pipefail

cd /root/drumblender || exit 1

WANDB_PROJECT="${WANDB_PROJECT:-drumblender}"
WANDB_NAME="${WANDB_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
WANDB_DIR="${WANDB_DIR:-/root/drumblender/logs/wandb}"
RUN_SEED="${RUN_SEED:-20260218}"

CFG="${CFG:-/root/drumblender/cfg/05_all_parallel.yaml}"
DATA_DIR="${DATA_DIR:-/root/datasets/modal_features/processed_modal_flat}"
CKPT_DIR="${CKPT_DIR:-/root/drumblender/ckpt}"
RESUME_CKPT="${RESUME_CKPT:-}"
MAX_EPOCHS="${MAX_EPOCHS:-125}"

NUM_DEVICES="${NUM_DEVICES:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-6}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-0.1}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-8}"
USE_BUCKETING="${USE_BUCKETING:-true}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

if [[ ! -f "$CFG" ]]; then
  echo "Config not found: $CFG"
  exit 1
fi

if [[ ! -f "$DATA_DIR/metadata.json" ]]; then
  echo "Dataset metadata not found: $DATA_DIR/metadata.json"
  exit 1
fi

# ### HIGHLIGHT: Always create output directories used by logger/checkpoint callbacks.
mkdir -p "$WANDB_DIR" "$CKPT_DIR" /root/drumblender/lightning_logs

CMD=(
  drumblender fit -c "$CFG"
  --seed_everything "$RUN_SEED"
  --trainer.accelerator gpu
  --trainer.devices "$NUM_DEVICES"
  --trainer.strategy ddp_find_unused_parameters_false
  --trainer.replace_sampler_ddp false
  --trainer.precision 32
  --trainer.max_epochs "$MAX_EPOCHS"
  --trainer.log_every_n_steps 40
  --trainer.num_sanity_val_steps 0
  --trainer.val_check_interval "$VAL_CHECK_INTERVAL"
  --trainer.limit_val_batches "$LIMIT_VAL_BATCHES"
  --trainer.default_root_dir /root/drumblender/lightning_logs
  --trainer.logger pytorch_lightning.loggers.WandbLogger
  --trainer.logger.init_args.project "$WANDB_PROJECT"
  --trainer.logger.init_args.name "$WANDB_NAME"
  --trainer.logger.init_args.save_dir "$WANDB_DIR"
  --trainer.logger.init_args.log_model false
  --data.class_path drumblender.data.AudioDataModule
  --data.data_dir "$DATA_DIR"
  --data.meta_file metadata.json
  --data.dataset_class drumblender.data.AudioWithParametersDataset
  --data.dataset_kwargs "{parameter_key: feature_file, split_strategy: sample_pack, expected_num_modes: 64, seed: $RUN_SEED}"
  --data.seed "$RUN_SEED"
  --data.sample_rate 48000
  --data.num_samples null
  --data.batch_size "$BATCH_SIZE"
  --data.num_workers "$NUM_WORKERS"
  --data.skip_prepare_data true
  --data.use_bucketing "$USE_BUCKETING"
  --data.bucket_boundaries "[48000, 96000, 192000, 384000]"
  --data.drop_last true
)

# ### HIGHLIGHT: Resume training from an explicit checkpoint when needed.
if [[ -n "$RESUME_CKPT" ]]; then
  CMD+=(--ckpt_path "$RESUME_CKPT")
fi

"${CMD[@]}"
