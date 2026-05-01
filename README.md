# bridge-rna — file structure and workflow

ExpressionBERT: masked gene expression prediction using a SLiMPerformer Transformer, trained on ARCHS4 bulk RNA-seq and evaluated zero-shot on NASA OSDR spaceflight data.

See `CLAUDE.md` for architecture, the species experiment, and the OSDR eval state.

## Directory layout

```
bridge-rna/
├── about.md                          ← you are here
├── CLAUDE.md                         project context: architecture, experiments, OSDR state
│
├── train_single.py                   train one variant (DDP), picks variant via DATASET_VARIANT env
├── train_moe.py                      MoE PoC: 3 frozen 5k experts + softmax gate — BLOCKED (see CLAUDE.md)
├── preprocessing.py                  streams ARCHS4 H5 from S3 → TPM → log1p → parquet batches
├── merge.py                          merges preprocessing batches → one expression.parquet
├── slim_performer_model.py           SLiMPerformer linear-attention layer (imports numerator_and_denominator)
├── numerator_and_denominator.py      feature-map math for SLiMPerformer (do not delete — imported)
│
├── evaluate_osdr.py                  zero-shot OSDR eval on a list of checkpoints
├── analyze_moe_headroom.py           training-free MoE analysis: oracle/grid/uniform vs. best single expert on OSDR
├── check_alignment.py                diagnostic for train/eval alignment bugs (see CLAUDE.md)
├── check_moe_gene_counts.py          diagnostic for the 3 5k experts' gene-vocab mismatch
├── prep_osdr_from_kmeng.py           one-shot: converts kmeng's preprocessed OSDR CSV → canonical parquet
├── fetch_reference_data.py           one-shot: populates data/ensembl and data/gencode reference files
│
├── scripts/                          Savio batch jobs (run from repo root: `sbatch scripts/...`)
│   ├── savio_preprocess.sh           job array, 3 variants, writes parquets to scratch
│   ├── savio_train_experiments.sh    job array, trains all 3 variants on savio2_1080ti
│   ├── savio_train_moe.sh            BLOCKED — needs the gene-alignment patch before it can run
│   ├── savio_analyze_moe_headroom.sh training-free MoE headroom analysis on OSDR (savio2 CPU, 4h)
│   └── savio_evaluate_osdr.sh        evaluates all 3 checkpoints on OSDR
│
├── configs/
│   ├── sweep_single_config.yaml      W&B sweep config for train_single.py
│   └── sweep_train_single.sh         sweep launcher
│
├── data/                             reference data (committed)
│   ├── ensembl/                        orthologs + protein-coding gene whitelist
│   ├── gencode/                        exon-length tables for TPM
│   └── osdr/                           OSDR metadata, ortholog map, cached osdr_expression.parquet
│
├── examples/                         analysis notebooks
│   ├── osdr_example.ipynb              walt's reference results (20jo1hdd) — benchmark to beat
│   ├── tcga_example.ipynb, 5b_tcga_analysis.ipynb, embeddings.ipynb
│
├── checkpoints_performer/            saved runs (incl. walt's 20jo1hdd — do not prune without care)
├── results/                          eval CSV/JSON output dir (populated by evaluate_osdr.py)
├── wandb/                            W&B run history
│
└── archive/                          stale/superseded — kept for reference, not active
    ├── train.py                      old non-single trainer (superseded by train_single.py)
    ├── savio_job.sh, savio_sweep_job.sh, sweep_config.yaml, sweep_train.sh
    ├── download_archs4.sh            obsolete — preprocessing streams from S3
    ├── test_archs4_url.py            one-off URL check
    ├── scratch/                      exploratory notebooks and old trainer drafts
    ├── tests/                        preprocessing_check.py one-offs (no current callers)
    ├── prepared_data/                token_id_mapping.csv (only referenced by archived scratch)
    └── results_5_epochs_5000_samples/  old run
```

## Workflow

Two paths: Savio (production) and local (diagnostics).

### Production (Savio)

1. **One-time reference data.** Run `python fetch_reference_data.py` to populate `data/ensembl/` and `data/gencode/`.
2. **Preprocess ARCHS4 → parquet** (streams from `mssm-seq-matrix/*_matrix_v11.h5` on S3; no local H5 download):
   ```bash
   sbatch scripts/savio_preprocess.sh
   ```
   Job array creates three dataset dirs: `data/archs4/human_5k_merged/`, `mouse_5k_merged/`, `mixed_5k_merged/`, each with `expression.parquet`.
3. **Train three variants** (job array, 4× GTX 1080 Ti per variant, W&B logging under project `bridge-rna`):
   ```bash
   sbatch --dependency=afterok:<preprocess_job_id> scripts/savio_train_experiments.sh
   ```
   Variant is chosen via `DATASET_VARIANT` env var set by the job array. Checkpoints land in `checkpoints/{variant}/best_model.pt`.
4. **Evaluate on OSDR zero-shot**:
   ```bash
   sbatch scripts/savio_evaluate_osdr.sh
   ```
   Writes `results/osdr_eval/osdr_eval_results.{csv,json}` and logs to W&B.

5. **MoE headroom (training-free) on OSDR** — does combining the 3 experts help?
   ```bash
   sbatch scripts/savio_analyze_moe_headroom.sh
   ```
   Runs the 3 frozen experts on OSDR in a shared common gene space (intersection of expert gene sets), then reports per-expert / uniform-1/3 / grid-best (3-simplex search) / oracle (per-sample best expert) Pearson, plus a high-variance-genes-only cut. Writes `results/moe_headroom_5k/{predictions.npz,report.json}`. The `oracle - best_single` gap is the empirical headroom for any future learned gate.

6. **(BLOCKED) MoE gate training** — `scripts/savio_train_moe.sh` exists but currently fails: the 3 expert checkpoints have different gene counts and column orders (see CLAUDE.md "MoE PoC Status"). Needs a canonical-↔-native alignment patch in `train_moe.py:load_expert` before it will run.

### Local diagnostic (current step per CLAUDE.md)

Before trusting the species comparison, confirm the models aren't hitting an alignment bug:

```bash
python check_alignment.py \
  --checkpoint checkpoints/human_5k/best_model.pt \
  --osdr-parquet data/osdr/osdr_expression.parquet
```

Runs 8 checks (gene order, config, input range, prediction range, gene-mean collapse, identity recovery, missing-gene mask fraction, present-only vs include-missing pearson). See CLAUDE.md for the full diagnostic plan.

## Conventions

- **Gene vocabulary** is `data/ensembl/protein_coding_ortholog_genes.txt` — every training and eval step must use this exact ordered list. `evaluate_osdr.py:load_canonical_genes()` is the one source of truth.
- **Normalization**: log1p(TPM). Set in `train_single.py` (`normalization=log1p_tpm`) and mirrored in OSDR preprocessing.
- **Mask token**: `-10`. Must match between training (`train_single.py:CONFIG['mask_token']`) and eval (`cfg['mask_token']` loaded from checkpoint).
- **Savio paths**: `/global/scratch/users/minggangli/bridge-rna/`, conda env `bridge-rna`, account `ic_cdss170`, partition `savio2_1080ti`.
