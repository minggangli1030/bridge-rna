#!/bin/bash
#SBATCH --job-name=moe-headroom-mixed
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2          # CPU node — SLiMPerformer OOMs on 1080Ti at 14k genes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/moe-headroom-mixed-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/moe-headroom-mixed-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

# Training-free MoE headroom on a MIXED mouse(OSDR)+human(TCGA) eval set.
# Single-species OSDR collapses oracle≈best_single by construction (every
# sample wants the same expert); a balanced mix forces routing variance,
# which is the actual headroom test before training a gate.

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
TCGA_PQ="data/tcga/tcga_expression.parquet"

N_EACH="${N_EACH:-500}"
SEED="${SEED:-42}"
MIXED_PQ="data/eval_mixed/mixed_${N_EACH}_${N_EACH}_seed${SEED}.parquet"
OUT_DIR="results/moe_headroom_5k_mixed_${N_EACH}_${N_EACH}_seed${SEED}"

for f in "$HUMAN_CKPT" "$MOUSE_CKPT" "$MIXED_CKPT" "$OSDR_PQ" "$TCGA_PQ"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required file: $f"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "Build mixed eval set"
echo "  n_each=$N_EACH  seed=$SEED"
echo "  output: $MIXED_PQ"
echo "========================================"

if [ ! -f "$MIXED_PQ" ]; then
    python build_mixed_eval.py \
        --osdr-parquet "$OSDR_PQ" \
        --tcga-parquet "$TCGA_PQ" \
        --n-each "$N_EACH" \
        --seed "$SEED" \
        --output "$MIXED_PQ"
else
    echo "Mixed parquet already exists; reusing."
fi

echo ""
echo "========================================"
echo "MoE headroom analysis (mixed eval)"
echo "  output: $OUT_DIR"
echo "========================================"

python analyze_moe_headroom.py \
    --osdr-parquet "$MIXED_PQ" \
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
    python -c "
import json
r = json.load(open('$OUT_DIR/report.json'))
print('  N_samples:', r['n_samples'], ' G_common:', r['n_common_genes'])
print()
print('  --- All common genes ---')
for x in r['all_common']:
    print(f\"    {x['name']:<35} {x['mean']:+.4f}\")
print()
if r.get('by_species'):
    print('  --- By species ---')
    for sp, d in r['by_species'].items():
        if sp == '__global__':
            continue
        per = d['per_expert_mean']
        print(f\"    {sp} (N={d['n']}): \"
              f\"h={per['human_5k']:+.4f}  m={per['mouse_5k']:+.4f}  x={per['mixed_5k']:+.4f}  \"
              f\"oracle={d['oracle_mean']:+.4f}  gap={d['gap_oracle_minus_best']:+.4f}  best={d['best_single']}\")
    g = r['by_species'].get('__global__')
    if g:
        print(f\"    GLOBAL  oracle={g['oracle_mean']:+.4f}  best_single={g['best_single_mean']:+.4f}  gap={g['gap_oracle_minus_best']:+.4f}\")
"
fi
