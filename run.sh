#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/drumblender}"
CFG_PATH="${CFG_PATH:-cfg/05_all_parallel.yaml}"
cd "$REPO_ROOT" || exit 1

LOSS_MODE="${1:-plain}"

case "$LOSS_MODE" in
  plain|baseline|off)
    RUN_PREFIX="run_"
    LOSS_CFG_PATH="$REPO_ROOT/cfg/loss/mss.yaml"
    ;;
  amp)
    RUN_PREFIX="run_AMP_"
    LOSS_CFG_PATH="$REPO_ROOT/cfg/loss/mss_log_rms.yaml"
    ;;
  smooth)
    RUN_PREFIX="run_SMOOTH_"
    LOSS_CFG_PATH="$REPO_ROOT/cfg/loss/mss_smoothl1.yaml"
    ;;
  si|legacy_si|on)
    RUN_PREFIX="run_SI_"
    LOSS_CFG_PATH="$REPO_ROOT/cfg/loss/safe_mss.yaml"
    ;;
  *)
    printf 'usage: bash run.sh [plain|amp|smooth|si|legacy_si]\n' >&2
    exit 1
    ;;
esac

WANDB_PROJECT="${WANDB_PROJECT:-drumblender}"
WANDB_NAME="${WANDB_NAME:-${RUN_PREFIX}$(date +%Y%m%d_%H%M%S)}"
WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/logs/wandb}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/train}"
RUN_LOG_DIR="${LOG_DIR}/${WANDB_NAME}"
RUN_LOG_FILE="${RUN_LOG_DIR}/train.log"
LIGHTNING_DIR="${RUN_LOG_DIR}/lightning"
RUN_CKPT_DIR="${REPO_ROOT}/ckpt/${WANDB_NAME}"

CKPT_PATH="${CKPT_PATH:-}"
LR="${LR:-}"

mkdir -p "$WANDB_DIR" "$RUN_LOG_DIR" "$LIGHTNING_DIR" "$RUN_CKPT_DIR"

CMD=(
  drumblender fit -c "$CFG_PATH"
  --model.init_args.loss_fn "$LOSS_CFG_PATH"
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
printf '[run.sh] loss mode: %s\n' "$LOSS_MODE"
printf '[run.sh] loss cfg: %s\n' "$LOSS_CFG_PATH"
printf '[run.sh] ckpt dir: %s\n' "$RUN_CKPT_DIR"
if [[ -n "$LR" ]]; then
  printf '[run.sh] lr override: %s\n' "$LR"
fi
"${CMD[@]}" 2>&1 | tee -a "$RUN_LOG_FILE"
