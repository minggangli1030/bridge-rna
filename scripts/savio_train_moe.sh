#!/bin/bash
#SBATCH --job-name=rna-train-moe
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/train-moe-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/train-moe-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# MoE proof-of-concept: train a small per-sample softmax gate over 3 frozen
# 5k experts (human_5k, mouse_5k, mixed_5k). Single-GPU is sufficient — only
# the gate gets gradients; experts run no_grad. ~12k combined train samples.
# Walltime budget 48h with comfortable margin.

set -eo pipefail
mkdir -p /global/scratch/users/minggangli/bridge-rna/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPUs: $CUDA_VISIBLE_DEVICES"
date

module purge
module load anaconda3/2024.02-1-11.4
source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate bridge-rna
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "Python: $(which python)"

export WANDB_PROJECT="bridge-rna"
export WANDB_ENTITY="minggangli1030"
export WANDB_RUN_NAME="moe_5k_v1"

cd /global/scratch/users/minggangli/bridge-rna

HUMAN_CKPT="checkpoints/human_5k/best_model.pt"
MOUSE_CKPT="checkpoints/mouse_5k/best_model.pt"
MIXED_CKPT="checkpoints/mixed_5k/best_model.pt"

for f in "$HUMAN_CKPT" "$MOUSE_CKPT" "$MIXED_CKPT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing expert checkpoint $f"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "MoE training (frozen 5k experts + gate)"
echo "  human: $HUMAN_CKPT"
echo "  mouse: $MOUSE_CKPT"
echo "  mixed: $MIXED_CKPT"
echo "========================================"

python train_moe.py \
    --human-ckpt "$HUMAN_CKPT" \
    --mouse-ckpt "$MOUSE_CKPT" \
    --mixed-ckpt "$MIXED_CKPT" \
    --checkpoint-dir checkpoints/moe_5k_v1 \
    --epochs 30 \
    --batch-size 4 \
    --lr 1e-3 \
    --mask-ratio 0.15

echo ""
echo "MoE training complete."
echo "Done at $(date)"
