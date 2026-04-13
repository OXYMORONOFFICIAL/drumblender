#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/drumblender}"
CFG_PATH="${CFG_PATH:-cfg/05_all_parallel.yaml}"
cd "$REPO_ROOT" || exit 1

LOSS_MODE="${1:-plain}"
NOISE_ENCODER_MODE="${2:-baseline}"
TRANSIENT_ENCODER_MODE="${3:-baseline}"

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
    printf 'usage: bash run.sh [plain|amp|smooth|si|legacy_si] [baseline|dac|dac_lstm] [baseline|dac]\n' >&2
    exit 1
    ;;
esac

case "$NOISE_ENCODER_MODE" in
  baseline|soundstream|off)
    NOISE_RUN_TAG=""
    NOISE_ENCODER_CFG_PATH=""
    ;;
  dac)
    NOISE_RUN_TAG="NOISEDAC_"
    NOISE_ENCODER_CFG_PATH="$REPO_ROOT/cfg/upgrades/encoders/noise_dac_style.yaml"
    ;;
  dac_lstm|daclstm|sequence)
    NOISE_RUN_TAG="NOISEDACLSTM_"
    NOISE_ENCODER_CFG_PATH="$REPO_ROOT/cfg/upgrades/encoders/noise_dac_lstm_style.yaml"
    ;;
  *)
    printf 'usage: bash run.sh [plain|amp|smooth|si|legacy_si] [baseline|dac|dac_lstm] [baseline|dac]\n' >&2
    exit 1
    ;;
esac

case "$TRANSIENT_ENCODER_MODE" in
  baseline|soundstream|off)
    TRANSIENT_RUN_TAG=""
    TRANSIENT_ENCODER_CFG_PATH=""
    ;;
  dac)
    TRANSIENT_RUN_TAG="TRANSDAC_"
    TRANSIENT_ENCODER_CFG_PATH="$REPO_ROOT/cfg/upgrades/encoders/transient_dac_style.yaml"
    ;;
  *)
    printf 'usage: bash run.sh [plain|amp|smooth|si|legacy_si] [baseline|dac|dac_lstm] [baseline|dac]\n' >&2
    exit 1
    ;;
esac

WANDB_PROJECT="${WANDB_PROJECT:-drumblender}"
WANDB_NAME="${WANDB_NAME:-${RUN_PREFIX}${NOISE_RUN_TAG}${TRANSIENT_RUN_TAG}$(date +%Y%m%d_%H%M%S)}"
WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/logs/wandb}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/train}"
RUN_LOG_DIR="${LOG_DIR}/${WANDB_NAME}"
RUN_LOG_FILE="${RUN_LOG_DIR}/train.log"
LIGHTNING_DIR="${RUN_LOG_DIR}/lightning"
RUN_CKPT_DIR="${REPO_ROOT}/ckpt/${WANDB_NAME}"

CKPT_PATH="${CKPT_PATH:-}"
LR="${LR:-}"

mkdir -p "$WANDB_DIR" "$RUN_LOG_DIR" "$LIGHTNING_DIR" "$RUN_CKPT_DIR"

if [[ -n "$NOISE_ENCODER_CFG_PATH" && ! -f "$NOISE_ENCODER_CFG_PATH" ]]; then
  printf 'noise encoder config not found: %s\n' "$NOISE_ENCODER_CFG_PATH" >&2
  exit 1
fi

if [[ -n "$TRANSIENT_ENCODER_CFG_PATH" && ! -f "$TRANSIENT_ENCODER_CFG_PATH" ]]; then
  printf 'transient encoder config not found: %s\n' "$TRANSIENT_ENCODER_CFG_PATH" >&2
  exit 1
fi

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

if [[ -n "$NOISE_ENCODER_CFG_PATH" ]]; then
  CMD+=(
    --model.init_args.noise_autoencoder "$NOISE_ENCODER_CFG_PATH"
    --model.init_args.noise_autoencoder_accepts_audio true
  )
fi

if [[ -n "$TRANSIENT_ENCODER_CFG_PATH" ]]; then
  CMD+=(
    --model.init_args.transient_autoencoder "$TRANSIENT_ENCODER_CFG_PATH"
    --model.init_args.transient_autoencoder_accepts_audio true
  )
fi

printf '[run.sh] log file: %s\n' "$RUN_LOG_FILE"
printf '[run.sh] wandb name: %s\n' "$WANDB_NAME"
printf '[run.sh] loss mode: %s\n' "$LOSS_MODE"
printf '[run.sh] loss cfg: %s\n' "$LOSS_CFG_PATH"
printf '[run.sh] noise encoder mode: %s\n' "$NOISE_ENCODER_MODE"
if [[ -n "$NOISE_ENCODER_CFG_PATH" ]]; then
  printf '[run.sh] noise encoder cfg: %s\n' "$NOISE_ENCODER_CFG_PATH"
fi
printf '[run.sh] transient encoder mode: %s\n' "$TRANSIENT_ENCODER_MODE"
if [[ -n "$TRANSIENT_ENCODER_CFG_PATH" ]]; then
  printf '[run.sh] transient encoder cfg: %s\n' "$TRANSIENT_ENCODER_CFG_PATH"
fi
printf '[run.sh] ckpt dir: %s\n' "$RUN_CKPT_DIR"
if [[ -n "$LR" ]]; then
  printf '[run.sh] lr override: %s\n' "$LR"
fi
"${CMD[@]}" 2>&1 | tee -a "$RUN_LOG_FILE"
