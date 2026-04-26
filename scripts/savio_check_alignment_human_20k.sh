#!/bin/bash
#SBATCH --job-name=align-check-20k
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/align-check-20k-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/align-check-20k-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# Alignment diagnostic for the human_20k checkpoint.
# Key check is [5]: corr(pred, gene_mean) vs corr(pred, true).
# - If they're close (e.g. ~0.81 each) the model is still gene-mean-collapsed,
#   just at a higher level. Need an architectural change.
# - If corr(pred, true) > corr(pred, gene_mean) the model is learning real
#   structure but not enough — diff config/data against Walt's setup.

set -eo pipefail
mkdir -p /global/scratch/users/minggangli/bridge-rna/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPUs: $CUDA_VISIBLE_DEVICES"
date

module purge
module load anaconda3/2024.02-1-11.4
source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate bridge-rna
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /global/scratch/users/minggangli/bridge-rna

python check_alignment.py \
    --checkpoint checkpoints/human_20k/best_model.pt \
    --osdr-parquet data/osdr/osdr_expression.parquet

echo ""
echo "Done at $(date)"
