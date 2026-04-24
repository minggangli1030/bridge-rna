#!/bin/bash
#SBATCH --job-name=rna-train-exp
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti   # GPU partition (2x GTX 1080 Ti per node)
#SBATCH --gres=gpu:4                # Request all 4 GPUs on the node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # 1 task per GPU for DDP
#SBATCH --cpus-per-task=2           # 4 tasks x 2 cpus = 8 total (node max)
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/train-exp-%A_%a.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/train-exp-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# ── Setup ──────────────────────────────────────────────────────────────────────
set -eo pipefail
mkdir -p /global/scratch/users/minggangli/bridge-rna/logs

echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

# ── Load modules ───────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.02-1-11.4

# ── Activate conda environment ─────────────────────────────────────────────────
source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate bridge-rna
echo "Python: $(which python)"

# ── Fix GLIBCXX version mismatch ───────────────────────────────────────────────
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── W&B authentication ─────────────────────────────────────────────────────────
export WANDB_PROJECT="bridge-rna"
export WANDB_ENTITY="minggangli1030"

# ── Project directory ──────────────────────────────────────────────────────────
cd /global/scratch/users/minggangli/bridge-rna

# ── Variant selection (indexed by SLURM_ARRAY_TASK_ID) ────────────────────────
#   0 → human_5k   : trained on 5k human samples
#   1 → mouse_5k   : trained on 5k mouse samples
#   2 → mixed_5k   : trained on 2.5k human + 2.5k mouse

VARIANTS=("human_5k" "mouse_5k" "mixed_5k")
export DATASET_VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo ""
echo "========================================"
echo "Training variant: $DATASET_VARIANT"
echo "========================================"

# ── Verify data exists ─────────────────────────────────────────────────────────
MERGED_PARQUET="data/archs4/${DATASET_VARIANT}_merged/expression.parquet"
if [ ! -f "$MERGED_PARQUET" ]; then
    echo "ERROR: $MERGED_PARQUET not found."
    echo "Run savio_preprocess.sh first to generate the datasets."
    exit 1
fi
echo "Data: $MERGED_PARQUET"

# ── Launch training ────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=4 \
    --master_port=$((29500 + SLURM_ARRAY_TASK_ID)) \
    train_single.py

echo ""
echo "========================================"
echo "Training complete for: $DATASET_VARIANT"
echo "========================================"
echo "Done at $(date)"
