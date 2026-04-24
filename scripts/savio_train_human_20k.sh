#!/bin/bash
#SBATCH --job-name=rna-train-20k
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/train-20k-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/train-20k-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# A/B scale-up: train human_20k (matches walt's ~20k scale) to test whether
# the 5k model's below-baseline OSDR perf was just undertraining.
# Expected: val_loss ~0.61 (walt) and OSDR per-sample pearson >0.85.

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

export DATASET_VARIANT="human_20k"
MERGED_PARQUET="data/archs4/human_20k_merged/expression.parquet"
if [ ! -f "$MERGED_PARQUET" ]; then
    echo "ERROR: $MERGED_PARQUET not found. Run savio_preprocess_human_20k.sh first."
    exit 1
fi
echo "Data: $MERGED_PARQUET"

echo ""
echo "========================================"
echo "Training variant: $DATASET_VARIANT"
echo "========================================"

torchrun \
    --nproc_per_node=4 \
    --master_port=29510 \
    train_single.py

echo ""
echo "Training complete for: $DATASET_VARIANT"
echo "Done at $(date)"
