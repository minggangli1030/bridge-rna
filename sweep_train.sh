#!/usr/bin/env bash
# Wrapper for wandb sweep to launch DDP training with torchrun
echo "[SWEEP] Starting training with torchrun..." >&2
exec torchrun --nproc_per_node=2 train.py "$@"
