#!/bin/bash
#SBATCH --job-name=rna-train-20k-v2
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/train-20k-v2-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/train-20k-v2-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# Architecture A/B on human_20k. Diagnostic on v1 (num_layers=2, mask_ratio=0.15,
# weight_decay=0) showed corr(pred, gene_mean) ≈ corr(pred, true) ≈ 0.76 — model
# collapsed to per-gene mean. v2 changes:
#   num_layers   2 → 4
#   mask_ratio   0.15 → 0.30
#   weight_decay 0 → 0.01
# Goal: corr(pred, true) > corr(pred, gene_mean) on the alignment diagnostic,
# and OSDR per-sample pearson > 0.85 (gene_mean baseline).
# Reuses the same human_20k merged parquet — no re-preprocessing.

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

cd /global/scratch/users/minggangli/bridge-rna

export DATASET_VARIANT="human_20k_v2"
MERGED_PARQUET="data/archs4/human_20k_merged/expression.parquet"
if [ ! -f "$MERGED_PARQUET" ]; then
    echo "ERROR: $MERGED_PARQUET not found."
    exit 1
fi
echo "Data: $MERGED_PARQUET"

echo ""
echo "========================================"
echo "Training variant: $DATASET_VARIANT"
echo "  num_layers:   4 (was 2)"
echo "  mask_ratio:   0.30 (was 0.15)"
echo "  weight_decay: 0.01 (was 0)"
echo "========================================"

torchrun \
    --nproc_per_node=4 \
    --master_port=29511 \
    train_single.py

echo ""
echo "Training complete for: $DATASET_VARIANT"
echo "Done at $(date)"
