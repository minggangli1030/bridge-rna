#!/bin/bash
#SBATCH --job-name=download-archs4
#SBATCH --account=ic_cdss170
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/global/scratch/users/minggangli/bridge-rna/logs/download-%j.out
#SBATCH --error=/global/scratch/users/minggangli/bridge-rna/logs/download-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=minggangli@berkeley.edu

set -eo pipefail
mkdir -p /global/scratch/users/minggangli/bridge-rna/logs

source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate bridge-rna

cd /global/scratch/users/minggangli/bridge-rna/data/archs4

pip install archs4py -q

echo "Downloading human_gene_v2.5.h5..."
python -c "import archs4py as a4; a4.data.download('human', version='v2.5', dest_dir='.')"

echo "Downloading mouse_gene_v2.5.h5..."
python -c "import archs4py as a4; a4.data.download('mouse', version='v2.5', dest_dir='.')"

echo "Download complete at $(date)"
ls -lh .
