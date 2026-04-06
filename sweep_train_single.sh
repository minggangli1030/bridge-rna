#!/usr/bin/env bash
# Wrapper for wandb sweep to launch DDP single-parquet training with torchrun
echo "[SWEEP] Starting single-parquet training with torchrun..." >&2

# Activate the correct conda environment so the right pandas/torch are used
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bridge-rna

# wandb agent sets CUDA_VISIBLE_DEVICES which breaks torchrun DDP — unset it
unset CUDA_VISIBLE_DEVICES
N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "[SWEEP] Detected $N_GPUS GPUs" >&2

exec torchrun --nproc_per_node=$N_GPUS --master_port=29500 \
  /global/scratch/users/minggangli/bridge-rna/train_single.py "$@"
