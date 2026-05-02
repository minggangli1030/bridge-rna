# bridge-rna

ExpressionBERT: masked gene-expression prediction with a SLiMPerformer (linear-attention) Transformer trained on ARCHS4 bulk RNA-seq.

## Architecture

- **Model**: `ExpressionPerformer` in `train_single.py` — gene-identity embeddings + Rotary Expression Embedding (REE) + SLiMPerformer layers + MSE reconstruction head
- **Objective**: MLM-style masking (mask N% of genes, predict their expression values)
- **Normalization**: log1p(TPM), computed in preprocessing
- **v1 arch** (collapsed): `num_layers=2`, `mask_ratio=0.15`, `weight_decay=0`
- **v2 arch** (current): `num_layers=4`, `mask_ratio=0.30`, `weight_decay=0.01`

## Data Pipeline

1. **`preprocessing.py`** — streams ARCHS4 from S3 (or reads local H5), QC + TPM + ortholog alignment, writes parquet batches.
2. **`merge.py`** — merges batches → single `expression.parquet`.
3. **`train_single.py`** — DDP training, variant selected via `DATASET_VARIANT` env var.

Reference data (committed under `data/ensembl/`, `data/gencode/`):
- `orthologs_one2one.txt` — Ensembl one-to-one mouse↔human orthologs
- `protein_coding_ortholog_genes.txt` — protein-coding gene whitelist
- `gencode_v49_gene_exon_lengths.csv` / `gencode_v49_mouse_gene_exon_lengths.csv` — exon lengths for TPM
- `canonical_genes_shared.txt` — **15,581-gene shared vocab** for v2 retrain (force-added; `data/` is gitignored)

Savio config: account `ic_cdss170`, partition `savio2_1080ti` (GPU) / `savio2` (CPU), conda env `bridge-rna`.

## Variants

| variant         | species          | samples         | vocab                  | arch | status                             |
|-----------------|------------------|-----------------|------------------------|------|------------------------------------|
| `human_5k`      | human            | 5,000           | per-variant (14,818)   | v1   | trained, gene-mean-collapsed       |
| `mouse_5k`      | mouse            | 5,000           | per-variant (14,562)   | v1   | trained, gene-mean-collapsed       |
| `mixed_5k`      | half human/mouse | 2,500 + 2,500   | per-variant (14,522)   | v1   | trained, gene-mean-collapsed       |
| `human_20k`     | human            | 20,000          | per-variant            | v1   | trained, also collapsed            |
| `human_20k_v2`  | human            | 20,000          | per-variant            | v2   | paused at ~50h ETA (~80h total)    |
| `human_5k_v2`   | human            | 5,000 → 4,301   | shared (15,581)        | v2   | **training (job 33978375)**        |
| `mouse_5k_v2`   | mouse            | 5,000           | shared (15,581)        | v2   | **training (job 33978375)**        |
| `mixed_5k_v2`   | half human/mouse | 2,500 + 2,500   | shared (15,581)        | v2   | **training (job 33978375)**        |

## Current State (2026-05-02)

Option 2 retrain in flight: 3 v2 variants training on Savio after v1 hit gene-mean collapse and the v1 MoE PoC was blocked by per-variant vocabs.

**Verified just now:**
- Preprocess job `33978374` completed (3 array tasks, ~1–2 min each).
- All 3 merged parquets have **identical 15,581-gene column order** — `cols[0] == cols[1] == cols[2]`. The alignment bug that broke the v1 MoE PoC is dead.
- Training job `33978375` running (4× 1080 Ti each, walltime cap 24h, expected ~4–8h per variant).

## Runbook (Option 2)

### Preprocess + train

```bash
cd /global/scratch/users/minggangli/bridge-rna
git pull

PRE=$(sbatch --parsable scripts/savio_preprocess_5k_v2.sh)
sbatch --dependency=afterok:$PRE scripts/savio_train_5k_v2.sh

# Monitor
squeue -u $USER -o "%.10i %.9P %.8j %.2t %.10M %.10L %R"
tail -f logs/train-5k-v2-<jobid>_<task>.out
# W&B: https://wandb.ai/minggangli1030/bridge-rna  (group by variant)
```

### Verify shared-vocab alignment after preprocess

```bash
conda activate bridge-rna
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -c "import pandas as pd; cols=[pd.read_parquet(f'data/archs4/{v}_merged/expression.parquet').columns.tolist() for v in ['human_5k_v2','mouse_5k_v2','mixed_5k_v2']]; print('lens:', [len(c) for c in cols]); print('all identical:', cols[0]==cols[1]==cols[2])"
# Expect: lens: [15581, 15581, 15581]   all identical: True
```

### After training: alignment diagnostic, then OSDR eval

`check_alignment.py` runs 8 sanity checks against a single checkpoint + the OSDR parquet. The keystone metric is **[5] `corr(pred, gene_mean)` vs `corr(pred, true)`** — if those are equal, the model has gene-mean-collapsed and is a strict downgrade of the baseline.

```bash
python check_alignment.py \
  --checkpoint checkpoints/human_5k_v2/best_model.pt \
  --osdr-parquet data/osdr/osdr_expression.parquet
```

Then OSDR eval on all 3 v2 checkpoints — point `evaluate_osdr.py --checkpoints` at `checkpoints/{variant}_v2/best_model.pt`. (No dedicated v2 eval script yet; reuse `scripts/savio_evaluate_osdr.sh` with the new paths or add `scripts/savio_eval_osdr_5k_v2.sh`.)

## Decision Rules

**Has the model learned anything?** OSDR per-sample Pearson > `baseline_gene_mean_pearson` (~0.85). v1 reference: walt's 20K model hits 0.892 at random_15pct.

| outcome on v2                                     | next step                                                                                                                                |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| no v2 variant clears 0.85                         | arch is the bottleneck even at v2 — try hidden_dim bump / LR schedule / longer warmup before scaling data                                |
| at least one v2 clears 0.85                       | rerun headroom analysis on v2 checkpoints (`analyze_moe_headroom.py`), then decide MoE                                                   |

**Does specialization exist?** Headroom = `oracle_per_sample − best_single_expert` Pearson, computed **per species** on a mixed eval set (OSDR mouse + TCGA human, via `build_mixed_eval.py`). Global gap is contaminated by species-ID and is uninterpretable on its own.

| per-species gap                  | conclusion                                                          |
|----------------------------------|---------------------------------------------------------------------|
| `≳ +0.02` on either sub-table    | learnable specialization exists → train the gate (`train_moe.py`)   |
| `≈ 0` on both sub-tables         | no headroom; experts uniformly weak — pivot, don't gate             |

## History

**2026-04-23** — v1 5k variants trained and eval'd zero-shot on 1855 OSDR mouse samples (`evaluate_osdr.py`). All three below gene_mean baseline:

| masking       | walt 20K | human_5k | mouse_5k | mixed_5k | gene_mean baseline |
|---------------|----------|----------|----------|----------|--------------------|
| random_15pct  | 0.892    | 0.687    | 0.781    | 0.758    | ~0.846             |
| random_50pct  | 0.842    | 0.685    | 0.770    | 0.757    | ~0.846             |
| block (b=.30) | 0.881    | 0.659    | 0.771    | 0.740    | ~0.840             |

**2026-04-24** — `check_alignment.py` on `human_5k`: `corr(pred, gene_mean) = 0.682 ≈ corr(pred, true) = 0.679`. **Gene-mean collapse confirmed.** Training-parquet vocab (14,818) ≠ canonical_genes (15,734); `evaluate_osdr.py:702-708` re-indexes correctly. Identity recovery at unmasked positions r≈0.86. Ranking (mouse > mixed > human) is real but meaningless inside the collapse regime.

**2026-04-30** — `human_20k_v2` training paused at ~50h ETA (~80h total). Pivoted to MoE PoC using existing 5k experts as frozen experts (`train_moe.py`).

**2026-05-01 (a)** — `train_moe.py` job `33939250` failed in <30s. `check_moe_gene_counts.py` revealed per-variant vocab divergence (14818 / 14562 / 14522). PoC's "shared gene space" assumption broken; `gene_embedding[i]` meant a different gene in each expert.

**2026-05-01 (b)** — Three options considered: (1) train-time alignment, (2) re-preprocess + retrain with shared vocab, (3) training-free headroom analysis. Picked (3) as the cheapest decision-informer. `analyze_moe_headroom.py` runs the 3 frozen experts on OSDR in a common gene space and reports per-expert / uniform / grid-best / oracle Pearson.

**2026-05-01 (c)** — First headroom run (job `33962526`, OSDR-only): oracle 0.7833 vs best_single 0.7822 → gap **+0.0011**. But OSDR is ~all mouse — every sample wants the same expert, so oracle ≈ best_single is structurally forced. Result is uninterpretable for the routing question.

**2026-05-02 (a)** — Mixed-eval headroom (job `33977614`, 500 OSDR mouse + 500 TCGA human):

| sub-table | best single        | oracle | gap         |
|-----------|--------------------|--------|-------------|
| human     | human_5k 0.7989    | 0.7992 | **+0.0003** |
| mouse     | mouse_5k 0.7822    | 0.7835 | **+0.0014** |
| global    | 0.7715             | 0.7914 | +0.0199     |

Per-species gaps both ≈ 0 → no learnable specialization at v1. Global +0.0199 is the species-ID artifact (a 2-line `if species == human` rule). Decision: pivot to option 2 — retrain with shared vocab under v2 arch.

**2026-05-02 (b)** — Option 2 implementation:
- `compute_shared_canonical.py` → `data/ensembl/canonical_genes_shared.txt` (15,581 genes; data-independent intersection of protein-coding-orthologs ∩ human-exon-lengths ∩ mouse-exon-lengths-via-map)
- `preprocessing.py` gained `--canonical-genes-file` — when provided, skips the data-dependent all-zero filter and uses the file verbatim (intersected with `valid_len_genes`)
- `human_5k_v2` / `mouse_5k_v2` / `mixed_5k_v2` added to `train_single.py:VARIANT_CONFIGS` with v2 arch
- `scripts/savio_preprocess_5k_v2.sh`, `scripts/savio_train_5k_v2.sh` — preprocess (job 33978374, completed) and train (job 33978375, running)
- Cross-variant column alignment verified: all 3 parquets have identical 15,581-gene order

## Blocked / Stale

- `train_moe.py` — needs the per-variant alignment patch OR re-aimed at v2 checkpoints (which now share a vocab and don't need patching). Don't run `scripts/savio_train_moe.sh` on v1 checkpoints; it'll fail in <30s.
- `human_20k_v2` — paused at ~50h ETA. Decision deferred until v2 5k results land.

## Files Quick Reference

| file                                           | purpose                                                                 |
|------------------------------------------------|-------------------------------------------------------------------------|
| `preprocessing.py`                             | extract + QC + TPM + canonical alignment; `--canonical-genes-file` flag |
| `merge.py`                                     | merge batch parquets                                                    |
| `train_single.py`                              | DDP training; `VARIANT_CONFIGS` selects dataset + arch overrides        |
| `compute_shared_canonical.py`                  | regenerate `canonical_genes_shared.txt`                                  |
| `check_alignment.py`                           | 8-check post-train diagnostic (run on a single ckpt + OSDR parquet)     |
| `evaluate_osdr.py`                             | zero-shot OSDR per-sample Pearson eval                                  |
| `analyze_moe_headroom.py`                      | training-free MoE headroom (per-expert / uniform / grid / oracle)        |
| `build_mixed_eval.py`, `prep_tcga.py`          | mixed OSDR+TCGA eval set construction                                   |
| `train_moe.py`                                 | gate training on frozen experts (BLOCKED on v1, retargetable to v2)     |
| `scripts/savio_preprocess_5k_v2.sh`            | array job — preprocess all 3 v2 variants                                |
| `scripts/savio_train_5k_v2.sh`                 | array job — train all 3 v2 variants                                     |
| `scripts/savio_evaluate_osdr.sh`               | OSDR eval (point `--checkpoints` at v2 paths to reuse)                   |
| `scripts/savio_analyze_moe_headroom_mixed.sh`  | mixed-eval headroom analysis                                            |
