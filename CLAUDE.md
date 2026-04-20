# bridge-rna

ExpressionBERT: masked gene expression prediction using a SLiMPerformer (linear attention) Transformer trained on ARCHS4 bulk RNA-seq data.

## Architecture

- **Model**: `ExpressionPerformer` in `train_single.py` — gene identity embeddings + Rotary Expression Embedding (REE) + SLiMPerformer layers + MSE reconstruction head
- **Objective**: MLM-style masking (mask 15% of genes, predict their expression values)
- **Normalization**: log1p(TPM), computed in preprocessing

## Data Pipeline

1. **`preprocessing.py`** — streams raw counts from ARCHS4 S3 (`mssm-seq-matrix/human_matrix_v11.h5`, `mouse_matrix_v11.h5`), applies QC, TPM normalization, ortholog alignment, outputs parquet batch files
2. **`merge.py`** — merges batch parquets into a single `expression.parquet` for training
3. **`train_single.py`** — trains on a single merged parquet with DDP

Reference data (committed under `data/ensembl/`, `data/gencode/`):
- `data/ensembl/orthologs_one2one.txt` — Ensembl one-to-one mouse↔human orthologs
- `data/ensembl/protein_coding_ortholog_genes.txt` — protein-coding gene whitelist
- `data/gencode/gencode_v49_gene_exon_lengths.csv` — human exon lengths for TPM
- `data/gencode/gencode_v49_mouse_gene_exon_lengths.csv` — mouse exon lengths for TPM

ARCHS4 H5 files are **not downloaded** — `preprocessing.py` streams directly from S3 at runtime.

## Species Experiment (3-variant comparison)

Three 5k-sample datasets to compare species-specific vs. cross-species generalization:

| Variant | Species | Samples |
|---------|---------|---------|
| `human_5k` | human only | 5,000 |
| `mouse_5k` | mouse only | 5,000 |
| `mixed_5k` | half human + half mouse | 2,500 + 2,500 |

### Running on Savio

```bash
# Preprocess all 3 variants (job array, streams from S3)
sbatch savio_preprocess.sh

# Train all 3 variants (job array, 4x GTX 1080 Ti, W&B logging)
sbatch --dependency=afterok:<preprocess_job_id> savio_train_experiments.sh
```

Training variant is selected via `DATASET_VARIANT` env var (set automatically by the job array). All 3 runs appear in W&B under project `bridge-rna`, grouped by variant name.

Savio config: account `ic_cdss170`, partition `savio2_1080ti`, conda env `bridge-rna`.

## Next Steps: Ensemble vs. MoE

Wait for W&B results from the 3-variant runs, then decide:

**Key signal to look for:**
- `human_5k` and `mouse_5k` val losses significantly lower than `mixed_5k` → species-specific specialization matters → **MoE**
- `mixed_5k` matches or beats single-species models → a single model generalizes fine → **Ensemble**

### Ensemble (low effort)
Average predictions from the 3 trained models at inference time. Since all models share the same gene space (shared orthologs), predictions are directly combinable. Simple post-training step, often gains a few points.

### Mixture of Experts (MoE, more interesting)
Add a gating network that routes each sample to the appropriate expert based on its expression profile. The 3 variant-trained models become the experts — the gate learns that human-like samples should route to the human expert, mouse-like to the mouse expert, etc.

Scientifically compelling question: *does the model learn species identity from expression alone?* The mixed dataset experiment is exactly the ablation that informs whether MoE routing is learnable.
