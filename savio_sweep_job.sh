#!/bin/bash
#SBATCH --job-name=bridge-rna-sweep
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti   # 2x GTX 1080 Ti per node
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/sweep-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/sweep-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

set -eo pipefail
mkdir -p /global/scratch/users/minggangli/bridge-rna/logs

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"
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

# ── W&B ────────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="Attention"
export WANDB_ENTITY="minggangli1030"

cd /global/scratch/users/minggangli/bridge-rna

# Pass sweep ID as argument: sbatch savio_sweep_job.sh <sweep-id>
SWEEP_ID=${1:-""}
if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: No sweep ID provided. Usage: sbatch savio_sweep_job.sh <sweep-id>"
  exit 1
fi

echo "Starting wandb agent for sweep: $SWEEP_ID"
wandb agent "minggangli1030/Attention/${SWEEP_ID}"

echo "Done at $(date)"
