#!/bin/bash
#SBATCH --job-name=rna-resume-20k-v2
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/resume-20k-v2-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/resume-20k-v2-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# Walltime-fallback resume for human_20k_v2.
# Auto-detects the latest epoch_*.pt across all run subdirs under
# checkpoints/human_20k_v2/, then continues training via RESUME_FROM env var.
# Override the checkpoint by setting RESUME_FROM before sbatch.

set -eo pipefail
mkdir -p /global/scratch/users/minggangli/bridge-rna/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPUs: $CUDA_VISIBLE_DEVICES"
date

module purge
module load anaconda3/2024.02-1-11.4
source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate bridge-rna
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export WANDB_PROJECT="bridge-rna"
export WANDB_ENTITY="minggangli1030"

cd /global/scratch/users/minggangli/bridge-rna

# Auto-detect the latest epoch_XX.pt under checkpoints/human_20k_v2/*/ if
# RESUME_FROM wasn't passed in. Filenames are zero-padded so lexical sort works.
if [ -z "${RESUME_FROM:-}" ]; then
    LATEST=$(ls -1 checkpoints/human_20k_v2/*/epoch_*.pt 2>/dev/null | sort | tail -1 || true)
    if [ -z "$LATEST" ]; then
        echo "ERROR: No epoch_*.pt found under checkpoints/human_20k_v2/. Run the v2 train script first."
        exit 1
    fi
    export RESUME_FROM="$LATEST"
fi
echo "RESUME_FROM: $RESUME_FROM"

export DATASET_VARIANT="human_20k_v2"
MERGED_PARQUET="data/archs4/human_20k_merged/expression.parquet"
[ -f "$MERGED_PARQUET" ] || { echo "ERROR: $MERGED_PARQUET not found"; exit 1; }

torchrun \
    --nproc_per_node=4 \
    --master_port=29512 \
    train_single.py

echo ""
echo "Resume run complete."
echo "Done at $(date)"
