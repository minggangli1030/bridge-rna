#!/bin/bash
#SBATCH --job-name=osdr-eval
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2_1080ti   # 1 GPU is enough for inference
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/osdr-eval-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/osdr-eval-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# ── Setup ──────────────────────────────────────────────────────────────────────
set -eo pipefail
mkdir -p /global/scratch/users/minggangli/bridge-rna/logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

# ── Modules + conda ────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.02-1-11.4
source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate bridge-rna
echo "Python: $(which python)"

# ── Fix GLIBCXX version mismatch ───────────────────────────────────────────────
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── W&B ────────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="bridge-rna"
export WANDB_ENTITY="minggangli1030"

# ── Project directory ──────────────────────────────────────────────────────────
cd /global/scratch/users/minggangli/bridge-rna

# ── Checkpoint paths ───────────────────────────────────────────────────────────
HUMAN_CKPT="checkpoints/human_5k/best_model.pt"
MOUSE_CKPT="checkpoints/mouse_5k/best_model.pt"
MIXED_CKPT="checkpoints/mixed_5k/best_model.pt"

# Verify at least one checkpoint exists
FOUND=0
for ckpt in "$HUMAN_CKPT" "$MOUSE_CKPT" "$MIXED_CKPT"; do
    if [ -f "$ckpt" ]; then
        echo "Checkpoint found: $ckpt"
        FOUND=$((FOUND + 1))
    else
        echo "WARNING: Checkpoint not found: $ckpt"
    fi
done

if [ "$FOUND" -eq 0 ]; then
    echo "ERROR: No checkpoints found. Run training first."
    exit 1
fi

# ── OSDR data ──────────────────────────────────────────────────────────────────
# Option 1: Use sp26_nasa raw data if available on scratch
SP26_RAW="/global/scratch/users/minggangli/sp26_nasa/data/osdr/raw"
METADATA="/global/scratch/users/minggangli/bridge-rna/data/osdr/metadata_new.csv"

# Copy metadata from sp26_nasa if not already copied
if [ ! -f "$METADATA" ]; then
    SP26_META="/global/scratch/users/minggangli/sp26_nasa/osdr_data/metadata_new.csv"
    SP26_ORTH="/global/scratch/users/minggangli/sp26_nasa/osdr_data/human_mouse_orthologs.csv"
    if [ -f "$SP26_META" ]; then
        mkdir -p "$(dirname "$METADATA")"
        cp "$SP26_META" "$METADATA"
        echo "Copied OSDR metadata from sp26_nasa"
        [ -f "$SP26_ORTH" ] && cp "$SP26_ORTH" "data/osdr/human_mouse_orthologs.csv"
    else
        # Fall back to local copy committed in repo
        METADATA="data/osdr/metadata_new.csv"
        echo "Using metadata from repo: $METADATA"
    fi
fi

# ── Build args ─────────────────────────────────────────────────────────────────
CKPT_ARGS=""
[ -f "$HUMAN_CKPT" ] && CKPT_ARGS="$CKPT_ARGS $HUMAN_CKPT"
[ -f "$MOUSE_CKPT" ] && CKPT_ARGS="$CKPT_ARGS $MOUSE_CKPT"
[ -f "$MIXED_CKPT" ] && CKPT_ARGS="$CKPT_ARGS $MIXED_CKPT"

RAW_DIR_ARG=""
if [ -d "$SP26_RAW" ]; then
    echo "Found sp26_nasa raw OSDR data at: $SP26_RAW"
    RAW_DIR_ARG="--osdr-raw-dir $SP26_RAW"
fi

# ── Run evaluation ─────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Running OSDR zero-shot evaluation"
echo "========================================"

python evaluate_osdr.py \
    --checkpoints $CKPT_ARGS \
    --output-dir results/osdr_eval \
    --metadata-csv "$METADATA" \
    $RAW_DIR_ARG \
    --batch-size 32 \
    --wandb

echo ""
echo "========================================"
echo "Evaluation complete"
echo "========================================"
echo "Done at $(date)"

# Show summary
if [ -f "results/osdr_eval/osdr_eval_results.csv" ]; then
    echo ""
    echo "Results CSV:"
    cat results/osdr_eval/osdr_eval_results.csv
fi
