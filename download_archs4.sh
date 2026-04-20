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

echo "Downloading human_matrix_v11.h5 from S3..."
python -c "
import s3fs, sys
s3 = s3fs.S3FileSystem(anon=True)
src = 'mssm-seq-matrix/human_matrix_v11.h5'
dst = 'human_matrix_v11.h5'
print('Starting human download...', flush=True)
s3.get(src, dst)
print('Human done.', flush=True)
"

echo "Downloading mouse_matrix_v11.h5 from S3..."
python -c "
import s3fs, sys
s3 = s3fs.S3FileSystem(anon=True)
src = 'mssm-seq-matrix/mouse_matrix_v11.h5'
dst = 'mouse_matrix_v11.h5'
print('Starting mouse download...', flush=True)
s3.get(src, dst)
print('Mouse done.', flush=True)
"

echo "Download complete at $(date)"
ls -lh .
