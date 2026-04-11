#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/drumblender}"
cd "$REPO_ROOT" || exit 1

WANDB_PROJECT="${WANDB_PROJECT:-drumblender}"
WANDB_NAME="${WANDB_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/logs/wandb}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/train}"
RUN_LOG_DIR="${LOG_DIR}/${WANDB_NAME}"
RUN_LOG_FILE="${RUN_LOG_DIR}/train.log"
LIGHTNING_DIR="${RUN_LOG_DIR}/lightning"

CKPT_PATH="${CKPT_PATH:-}"
LR="${LR:-}"

mkdir -p "$WANDB_DIR" "$RUN_LOG_DIR" "$LIGHTNING_DIR"

CMD=(
  drumblender fit -c cfg/05_all_parallel.yaml
  --trainer.default_root_dir "$LIGHTNING_DIR"
  --trainer.logger pytorch_lightning.loggers.WandbLogger
  --trainer.logger.init_args.project "$WANDB_PROJECT"
  --trainer.logger.init_args.name "$WANDB_NAME"
  --trainer.logger.init_args.save_dir "$WANDB_DIR"
  --trainer.logger.init_args.log_model false
)

if [[ -n "$CKPT_PATH" ]]; then
  CMD+=(--ckpt_path "$CKPT_PATH")
fi

if [[ -n "$LR" ]]; then
  CMD+=(--optimizer.init_args.lr "$LR")
fi

printf '[run.sh] log file: %s\n' "$RUN_LOG_FILE"
printf '[run.sh] wandb name: %s\n' "$WANDB_NAME"
if [[ -n "$LR" ]]; then
  printf '[run.sh] lr override: %s\n' "$LR"
fi
"${CMD[@]}" 2>&1 | tee -a "$RUN_LOG_FILE"
