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
sbatch scripts/savio_preprocess.sh

# Train all 3 variants (job array, 4x GTX 1080 Ti, W&B logging)
sbatch --dependency=afterok:<preprocess_job_id> scripts/savio_train_experiments.sh
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

## Phase 2: Benchmarking on OSDR (NASA Spaceflight Data)

After training completes, evaluate all 3 models **zero-shot** on OSDR data — no retraining, just forward pass.

**Why OSDR is a strong benchmark:**
- True OOD test: microgravity, radiation, spaceflight stress responses — nothing like ARCHS4 training data
- Controlled comparison: same gene space (shared orthologs), same preprocessing, only training data differs
- Both human and mouse samples available → all 3 variants can be evaluated fairly
- Tests whether learned gene-gene relationships transfer to genuinely new biology, not just held-out ARCHS4 patterns

**The benchmark story:**
> "We train on ARCHS4 bulk RNA-seq and evaluate zero-shot reconstruction on NASA spaceflight data — a domain the model never saw during training."

**What to measure:**
- Reconstruction loss (MSE on masked genes) per model on OSDR samples
- Does `mixed_5k` outperform `human_5k` on human spaceflight samples? → cross-species training generalizes better
- Does `human_5k` beat `mixed_5k` on human samples? → in-distribution specialization wins

**Key question this answers:**
The OSDR benchmark directly informs the Ensemble vs. MoE decision:
- `mixed_5k` best on OSDR → model learns fundamental expression biology → Ensemble likely sufficient
- Single-species models best on matched-species OSDR → specialization matters → MoE worth building

**Practical notes:**
- OSDR datasets are small (tens to hundreds of samples per study) — aggregate across multiple studies for stable signal
- `sp26_nasa` repo already has some OSDR preprocessing infrastructure to build on
- Preprocessing must use the same gene vocabulary (`canonical_genes.csv`) as the trained models

## OSDR Eval — Current State (2026-04-23)

All 3 variants trained and evaluated zero-shot on OSDR via `evaluate_osdr.py` (n=1855 mouse samples). Reported metric is **per-sample Pearson on masked positions** (see `pearson_per_sample`).

| masking       | walt (20K-sample ref) | human_5k | mouse_5k | mixed_5k | gene_mean baseline |
|---------------|-----------------------|----------|----------|----------|--------------------|
| random_15pct  | 0.892                 | 0.687    | 0.781    | 0.758    | ~0.846             |
| random_50pct  | 0.842                 | 0.685    | 0.770    | 0.757    | ~0.846             |
| block         | 0.881 (b=0.30)        | 0.659    | 0.771    | 0.740    | ~0.840             |

**Problem:** all three variants are **below the gene_mean baseline**. Walt's reference model clears it by ~5 points. Species-comparison conclusions (mouse>mixed>human; MoE vs Ensemble) are not yet interpretable — the models haven't learned anything beyond marginal gene statistics, so the ranking is unreliable.

**Do not start Ensemble/MoE work until at least one variant beats `baseline_gene_mean_pearson`.**

### Diagnostic plan (in order)

**Step 1 — Rule out an alignment bug first (cheap).** `check_alignment.py` at repo root runs 8 checks on a single checkpoint + OSDR parquet:
  1. checkpoint config dump
  2. train-parquet gene order vs `canonical_genes`
  3. OSDR input distribution (expect log1p TPM, ~0–10)
  4. prediction spot-check (range, per-sample pearson, constant-output flag)
  5. corr(pred, gene_mean) — tests if the model collapsed to "predict gene average"
  6. identity-recovery at UNmasked positions (if low → model is broken independent of masking)
  7. fraction of random-mask positions landing on zero-filled (OSDR-missing) genes
  8. pearson split: present-only genes vs include-missing (known concern: the mask at `evaluate_osdr.py:449` samples from all `G` genes, including the ~fill-zero missing ones)

Run:
```bash
python check_alignment.py \
  --checkpoint checkpoints/human_5k/best_model.pt \
  --osdr-parquet data/osdr/osdr_expression.parquet
```

**Step 2 — If alignment is clean, scale up `human_5k` to 20K samples** (matches Walt's scale) as a direct A/B. If the 20K run beats the gene_mean baseline, scale is the issue and the 3-variant experiment needs redoing at 20K. If it still fails → architecture/training-loop bug.

## Alignment Diagnostic Results (2026-04-24)

Ran `check_alignment.py` on `checkpoints/human_5k/best_model.pt` (via GPU srun on savio2_1080ti). Alignment is fine — the model is genuinely undertrained.

**Key finding from [5]:** `corr(pred, gene_mean) = 0.682` ≈ `corr(pred, true) = 0.679`. The model's *entire* signal comes from predicting the per-gene mean. It's a strictly noisier version of the gene_mean baseline, which is exactly why it loses to it.

Other checks:
- **[2]** Training parquet has 14818 genes, canonical_genes has 15734 — `evaluate_osdr.py:702-708` handles this via re-indexing (confirmed firing during real eval; otherwise it would have crashed at `gene_embedding`).
- **[4]** Predictions calibrated on first-order stats (per-sample mean 2.9 ≈ truth 2.9, std close). Model isn't broken, just shallow.
- **[6]** Identity recovery at unmasked positions r=0.86–0.87 — OK, not 1.0 but not zero.
- **[7]/[8]** After re-alignment, all 14818 training genes are present in OSDR (delta = 0.000). The "mask lands on zero-filled gene" concern doesn't apply.

**Decision:** proceed with Step 2. Ranking (mouse>mixed>human) in the 5K results CSV is real but meaningless — all three are in the gene-mean-collapse regime.

## Step 2 — human_20k A/B scale-up

Dedicated scripts for the one-variant A/B (leaves `human_5k`/`mouse_5k`/`mixed_5k` intact):

```bash
# 1. Preprocess 20k human samples (~6-8h on savio2)
sbatch scripts/savio_preprocess_human_20k.sh

# 2. Train after preprocess completes (24h wall on 4x 1080 Ti)
sbatch --dependency=afterok:<preprocess_jobid> scripts/savio_train_human_20k.sh

# 3. Evaluate on OSDR (add --checkpoints checkpoints/human_20k/best_model.pt when running evaluate_osdr.py)
```

`train_single.py` has `human_20k` in `VARIANT_CONFIGS`: `train_subset=16000`, `val_subset=3200`, balanced_sampling off.

**Success criterion:** OSDR per-sample Pearson > `baseline_gene_mean_pearson` (~0.85). Walt's 20K model hits 0.892 at random_15pct — that's the target. If `human_20k` also stays at ~0.68, the issue is architectural (e.g., 2 layers insufficient, mask ratio 0.15 wrong for 14k-token sequence, LR schedule) not data scale.
