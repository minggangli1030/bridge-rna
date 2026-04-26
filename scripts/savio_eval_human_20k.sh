#!/bin/bash
#SBATCH --job-name=osdr-eval-20k
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/osdr-eval-20k-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/osdr-eval-20k-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# OSDR zero-shot evaluation for the human_20k checkpoint (epoch 28, val_loss 0.317).
# Pass criterion: pearson_per_sample at random_15pct > ~0.85 (gene_mean baseline).
# Walt's reference is 0.892.

set -eo pipefail
mkdir -p /global/scratch/users/minggangli/bridge-rna/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
date

module purge
module load anaconda3/2024.02-1-11.4
source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate bridge-rna
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB_PROJECT=bridge-rna
export WANDB_ENTITY=minggangli1030

cd /global/scratch/users/minggangli/bridge-rna

CKPT="checkpoints/human_20k/best_model.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found"
    exit 1
fi
echo "Checkpoint: $CKPT"

python evaluate_osdr.py \
    --checkpoints "$CKPT" \
    --output-dir results/osdr_eval_human_20k \
    --metadata-csv data/osdr/metadata_new.csv \
    --osdr-parquet data/osdr/osdr_expression.parquet \
    --batch-size 16 \
    --device cpu \
    --wandb

echo ""
echo "Done at $(date)"
if [ -f results/osdr_eval_human_20k/osdr_eval_results.csv ]; then
    echo ""
    echo "Results CSV:"
    cat results/osdr_eval_human_20k/osdr_eval_results.csv
fi
