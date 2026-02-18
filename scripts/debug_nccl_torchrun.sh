#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

torchrun --standalone --nnodes=1 --nproc_per_node=2 -m scripts.debug_torch_distributed_nccl

