#!/bin/bash
#SBATCH --job-name=rna-train-5k-v2
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/train-5k-v2-%A_%a.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/train-5k-v2-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# Option-2 retrain. Each variant uses:
#   - shared canonical vocab (15581 genes, identical column order across variants)
#   - v2 architecture: num_layers=4, mask_ratio=0.30, weight_decay=0.01
# Same v2 changes that human_20k_v2 uses. Goal: at least one variant clears
# the gene-mean baseline (~0.85 OSDR per-sample Pearson) AND the species-
# specific variants beat mixed on matched-species OSDR — that's the signal
# MoE needs. If all three still collapse to gene-mean, the architecture is
# the bottleneck and 5k scale alone won't fix it.

set -eo pipefail
mkdir -p /global/scratch/users/minggangli/bridge-rna/logs

echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME  GPUs: $CUDA_VISIBLE_DEVICES"
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

VARIANTS=("human_5k_v2" "mouse_5k_v2" "mixed_5k_v2")
export DATASET_VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo ""
echo "========================================"
echo "Training variant: $DATASET_VARIANT"
echo "  num_layers:   4 (was 2)"
echo "  mask_ratio:   0.30 (was 0.15)"
echo "  weight_decay: 0.01 (was 0)"
echo "  vocab:        shared canonical (15581 genes)"
echo "========================================"

MERGED_PARQUET="data/archs4/${DATASET_VARIANT}_merged/expression.parquet"
if [ ! -f "$MERGED_PARQUET" ]; then
    echo "ERROR: $MERGED_PARQUET not found."
    echo "Run savio_preprocess_5k_v2.sh first."
    exit 1
fi
echo "Data: $MERGED_PARQUET"

torchrun \
    --nproc_per_node=4 \
    --master_port=$((29520 + SLURM_ARRAY_TASK_ID)) \
    train_single.py

echo ""
echo "========================================"
echo "Training complete for: $DATASET_VARIANT"
echo "========================================"
echo "Done at $(date)"
