#!/bin/bash
#SBATCH --job-name=bridge-rna
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti   # GPU partition (2x GTX 1080 Ti per node)
#SBATCH --gres=gpu:4                # Request all 4 GPUs on the node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # 1 task per GPU for DDP
#SBATCH --cpus-per-task=2           # 4 tasks x 2 cpus = 8 total (node max)
#SBATCH --mem=60G
#SBATCH --time=12:00:00             # Adjust as needed (max 72h on savio2_gpu)
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/bridge-rna-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/bridge-rna-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu   # <-- CHANGE to your email

# ── Setup ──────────────────────────────────────────────────────────────────────
set -eo pipefail
mkdir -p /global/scratch/users/minggangli/bridge-rna/logs

echo "Job ID: $SLURM_JOB_ID"
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
# Set your W&B API key (get it from https://wandb.ai/authorize)
# W&B credentials are stored in ~/.netrc from `wandb login`
export WANDB_PROJECT="bridge-rna"
export WANDB_ENTITY="minggangli1030"

# ── Project directory ──────────────────────────────────────────────────────────
cd /global/scratch/users/minggangli/bridge-rna

# ── Launch training ────────────────────────────────────────────────────────────
torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  train_single.py

echo "Done at $(date)"
