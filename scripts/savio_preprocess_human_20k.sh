#!/bin/bash
#SBATCH --job-name=rna-preprocess-20k
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/preprocess-20k-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/preprocess-20k-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# A/B scale-up: preprocess 20k human ARCHS4 samples for direct comparison
# against Walt's ~20k-sample model.

set -eo pipefail
mkdir -p /global/scratch/users/minggangli/bridge-rna/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
date

module purge
module load anaconda3/2024.02-1-11.4
source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate bridge-rna
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "Python: $(which python)"

cd /global/scratch/users/minggangli/bridge-rna

VARIANT="human_20k"
PREPROCESS_DIR="data/archs4/${VARIANT}"
MERGED_DIR="data/archs4/${VARIANT}_merged"

echo ""
echo "========================================"
echo "Variant:     $VARIANT"
echo "Species:     human"
echo "Max samples: 20000"
echo "========================================"

echo ""
echo "[STEP 1] Preprocessing (streaming from S3)..."
python preprocessing.py \
    --species human \
    --max-samples 20000 \
    --output-dir "$PREPROCESS_DIR" \
    --normalization tpm \
    --gene-set shared_orthologs \
    --qc-min-nonzero 14000

echo "[STEP 1] Done."

echo ""
echo "[STEP 2] Merging batch files..."
python merge.py \
    --input-dir "$PREPROCESS_DIR" \
    --output-dir "$MERGED_DIR"

echo "[STEP 2] Done."
echo ""
echo "Preprocessing complete: $MERGED_DIR/expression.parquet"
echo "Done at $(date)"
