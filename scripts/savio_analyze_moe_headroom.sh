#!/bin/bash
#SBATCH --job-name=moe-headroom
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2          # CPU node — SLiMPerformer OOMs on 1080Ti at 14k genes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/moe-headroom-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/moe-headroom-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# Training-free MoE headroom analysis: runs the 3 frozen 5k experts on OSDR
# in a shared common gene space, computes per-expert / uniform / grid-best /
# oracle Pearson on masked positions. PoC of whether MoE has measurable
# headroom over the best single expert before any gate is trained.

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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

HUMAN_CKPT="checkpoints/human_5k/best_model.pt"
MOUSE_CKPT="checkpoints/mouse_5k/best_model.pt"
MIXED_CKPT="checkpoints/mixed_5k/best_model.pt"
OSDR_PQ="data/osdr/osdr_expression.parquet"

for f in "$HUMAN_CKPT" "$MOUSE_CKPT" "$MIXED_CKPT" "$OSDR_PQ"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required file: $f"
        exit 1
    fi
done

OUT_DIR="results/moe_headroom_5k"

echo ""
echo "========================================"
echo "MoE headroom analysis"
echo "  output: $OUT_DIR"
echo "========================================"

python analyze_moe_headroom.py \
    --osdr-parquet "$OSDR_PQ" \
    --human-ckpt "$HUMAN_CKPT" \
    --mouse-ckpt "$MOUSE_CKPT" \
    --mixed-ckpt "$MIXED_CKPT" \
    --output-dir "$OUT_DIR" \
    --mask-ratio 0.15 \
    --batch-size 16 \
    --device cpu \
    --high-var-topk 1000

echo ""
echo "Done at $(date)"

if [ -f "$OUT_DIR/report.json" ]; then
    echo ""
    echo "Report summary:"
    python -c "import json; r=json.load(open('$OUT_DIR/report.json')); \
print('  N_samples:', r['n_samples'], ' G_common:', r['n_common_genes']); \
print('  --- All common genes ---'); \
[print(f\"  {x['name']:<35} {x['mean']:+.4f}\") for x in r['all_common']]; \
print('  --- High-variance subset ---'); \
[print(f\"  {x['name']:<35} {x['mean']:+.4f}\") for x in r['high_variance']['rows']]"
fi
