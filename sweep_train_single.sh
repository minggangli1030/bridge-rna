#!/usr/bin/env bash
# Wrapper for wandb sweep to launch DDP single-parquet training with torchrun
echo "[SWEEP] Starting single-parquet training with torchrun..." >&2
exec torchrun --nproc_per_node=4 --master_port=29500 \
  /global/scratch/users/minggangli/bridge-rna/train_single.py "$@"
