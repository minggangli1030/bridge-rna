#!/bin/bash
#SBATCH --job-name=rna-preprocess
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/preprocess-%A_%a.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/preprocess-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# ── Setup ──────────────────────────────────────────────────────────────────────
set -eo pipefail
mkdir -p /global/scratch/users/minggangli/bridge-rna/logs

echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
date

# ── Load modules ───────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.02-1-11.4

# ── Activate conda environment ─────────────────────────────────────────────────
source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate bridge-rna
echo "Python: $(which python)"

# ── Project directory ──────────────────────────────────────────────────────────
cd /global/scratch/users/minggangli/bridge-rna

# ── Variant definitions (indexed by SLURM_ARRAY_TASK_ID) ──────────────────────
#   0 → human_5k   : 5k human samples
#   1 → mouse_5k   : 5k mouse samples
#   2 → mixed_5k   : 2.5k human + 2.5k mouse = 5k total

VARIANTS=("human_5k" "mouse_5k" "mixed_5k")
SPECIES=("human" "mouse" "both")
MAX_SAMPLES=(5000 5000 2500)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SP=${SPECIES[$SLURM_ARRAY_TASK_ID]}
N=${MAX_SAMPLES[$SLURM_ARRAY_TASK_ID]}

PREPROCESS_DIR="data/archs4/${VARIANT}"
MERGED_DIR="data/archs4/${VARIANT}_merged"

echo ""
echo "========================================"
echo "Variant:     $VARIANT"
echo "Species:     $SP"
echo "Max samples: $N (per species)"
echo "Output dir:  $PREPROCESS_DIR"
echo "Merged dir:  $MERGED_DIR"
echo "========================================"

# ── Step 1: Preprocess ────────────────────────────────────────────────────────
echo ""
echo "[STEP 1] Running preprocessing..."
python preprocessing.py \
    --species "$SP" \
    --max-samples "$N" \
    --output-dir "$PREPROCESS_DIR" \
    --normalization tpm \
    --gene-set shared_orthologs \
    --qc-min-nonzero 14000

echo "[STEP 1] Done."

# ── Step 2: Merge batch files into single expression.parquet ──────────────────
echo ""
echo "[STEP 2] Merging batch files..."
python merge.py \
    --input-dir "$PREPROCESS_DIR" \
    --output-dir "$MERGED_DIR"

echo "[STEP 2] Done."

echo ""
echo "========================================"
echo "Preprocessing complete for: $VARIANT"
echo "  Batch files: $PREPROCESS_DIR/batch_files/"
echo "  Merged:      $MERGED_DIR/expression.parquet"
echo "========================================"
echo "Done at $(date)"
