#!/bin/bash
#SBATCH --job-name=osdr-eval-moe
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/osdr-eval-moe-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/osdr-eval-moe-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# Zero-shot OSDR eval for the MoE gate (frozen 5k experts + softmax gate).
# Compare metrics (random_15pct / random_50pct / block_50) against the best
# single 5k expert (mouse_5k @ 0.781 random_15pct).

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

GATE="checkpoints/moe_5k_v1/best_gate.pt"
if [ ! -f "$GATE" ]; then
    echo "ERROR: $GATE not found"
    exit 1
fi
echo "Gate: $GATE"

python evaluate_osdr_moe.py \
    --gate "$GATE" \
    --output-dir results/osdr_eval_moe_5k_v1 \
    --metadata-csv data/osdr/metadata_new.csv \
    --osdr-parquet data/osdr/osdr_expression.parquet \
    --variant-name moe_5k_v1 \
    --batch-size 16 \
    --device cpu \
    --wandb

echo ""
echo "Done at $(date)"
if [ -f results/osdr_eval_moe_5k_v1/osdr_eval_moe_results.csv ]; then
    echo ""
    echo "Results CSV:"
    cat results/osdr_eval_moe_5k_v1/osdr_eval_moe_results.csv
fi
