#!/bin/bash
# One-shot runner for the mixed-species MoE headroom eval on Savio.
#
# Usage (from Savio login node):
#     cd /global/scratch/users/minggangli/bridge-rna
#     bash scripts/run_mixed_eval_on_savio.sh
#
# Optional env overrides:
#     N_EACH=750 SEED=7 bash scripts/run_mixed_eval_on_savio.sh
#
# What it does:
#   1. git pull (latest code).
#   2. Checks the TCGA expression parquet exists; if not, prints the exact
#      scp command to upload it from your laptop and exits.
#   3. Submits the sbatch job.

set -eo pipefail

REPO=/global/scratch/users/minggangli/bridge-rna
cd "$REPO"

echo "==> git pull"
git pull --ff-only

mkdir -p data/tcga data/eval_mixed logs

TCGA_PQ="data/tcga/tcga_expression.parquet"
OSDR_PQ="data/osdr/osdr_expression.parquet"
CKPTS=(checkpoints/human_5k/best_model.pt
       checkpoints/mouse_5k/best_model.pt
       checkpoints/mixed_5k/best_model.pt)

echo ""
echo "==> Checking prerequisites"
missing=0
for f in "$TCGA_PQ" "$OSDR_PQ" "${CKPTS[@]}"; do
    if [ -f "$f" ]; then
        echo "  ok  $f"
    else
        echo "  MISSING  $f"
        missing=1
    fi
done

if [ ! -f "$TCGA_PQ" ]; then
    echo ""
    echo "TCGA parquet is missing. Upload from your laptop:"
    echo ""
    echo "  scp /Users/minggangli/Workspace/bridge-rna/data/tcga/tcga_expression.parquet \\"
    echo "      dtn.brc.berkeley.edu:$REPO/data/tcga/"
    echo ""
    echo "Then re-run this script."
fi

if [ "$missing" -ne 0 ]; then
    exit 1
fi

echo ""
echo "==> Submitting sbatch (N_EACH=${N_EACH:-500}, SEED=${SEED:-42})"
sbatch scripts/savio_analyze_moe_headroom_mixed.sh
echo ""
echo "Watch the queue:   squeue -u \$USER"
echo "Logs land in:      logs/moe-headroom-mixed-<jobid>.{out,err}"
