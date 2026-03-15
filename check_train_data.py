"""
Validation: pick a random sample from the processed parquet,
fetch the same sample from raw ARCHS4 H5, intersect genes,
compute TPM on both, and check correlation.
"""

import json
import numpy as np
import pandas as pd
import h5py
import archs4py as a4

# --- Load processed data ---
parquet_path = "data/archs4/train_orthologs/expression.parquet"
meta_path = "data/archs4/train_orthologs/metadata.csv"

expr = pd.read_parquet(parquet_path)  # (genes, samples)
meta = pd.read_csv(meta_path)

# Pick a random sample
rng = np.random.RandomState(123)
idx = rng.randint(len(meta))
sample_id = meta.loc[idx, "geo_accession"]
species = meta.loc[idx, "species"]
print(f"Sample: {sample_id} ({species}), index {idx}")

processed_vals = expr[sample_id]  # Series, index = gene symbols (human)

# --- Fetch raw counts from ARCHS4 ---
h5_file = f"data/archs4/{'human' if species == 'human' else 'mouse'}_gene_v2.5.h5"
print(f"Opening {h5_file}...")

geo_acc = a4.data.samples(h5_file)
match = np.where(np.array(geo_acc) == sample_id)[0]
if len(match) == 0:
    raise ValueError(f"{sample_id} not found in H5!")
h5_idx = match[0]
print(f"Found at H5 column index {h5_idx}")

with h5py.File(h5_file, "r") as h:
    gene_symbols = np.array([g.decode() for g in h["meta/genes/symbol"][:]])
    raw_counts = h["data/expression"][:, h5_idx].astype(np.float64)

raw_df = pd.Series(raw_counts, index=gene_symbols, name=sample_id)
# Aggregate duplicate gene symbols
raw_df = raw_df.groupby(level=0).sum()

# --- Compute TPM from raw counts (same logic as preprocessing.py) ---
if species == "human":
    lengths = pd.read_csv("data/gencode/gencode_v49_gene_exon_lengths.csv")
else:
    lengths = pd.read_csv("data/gencode/gencode_v49_mouse_gene_exon_lengths.csv")
lengths = lengths.set_index("gene_symbol")["exon_length"]

# Keep genes with exon lengths
common_raw = raw_df.index.intersection(lengths.index)
raw_df = raw_df.loc[common_raw]
lengths_aligned = lengths.loc[common_raw]

rate = raw_df / (lengths_aligned / 1000.0)
tpm_raw = rate / rate.sum() * 1e6
tpm_raw.name = "tpm_from_h5"

# --- If mouse, remap to human gene symbols ---
if species == "mouse":
    ortho = pd.read_csv("data/ensembl/orthologs_one2one.txt", sep="\t")
    m2h = dict(zip(ortho["Gene name"], ortho["Human gene name"]))
    tpm_raw.index = [m2h.get(g, g) for g in tpm_raw.index]
    tpm_raw = tpm_raw.groupby(level=0).sum()

# --- Intersect genes ---
shared_genes = processed_vals.index.intersection(tpm_raw.index)
# Drop genes that are zero in both
proc = processed_vals.loc[shared_genes]
raw = tpm_raw.loc[shared_genes]
nonzero = (proc > 0) | (raw > 0)
proc = proc[nonzero]
raw = raw[nonzero]

print(f"\nShared genes: {len(shared_genes):,}")
print(f"Non-zero in at least one: {len(proc):,}")

# --- Correlation ---
corr_pearson = proc.corr(raw)
corr_spearman = proc.rank().corr(raw.rank())

print(f"\nPearson correlation:  {corr_pearson:.6f}")
print(f"Spearman correlation: {corr_spearman:.6f}")

# --- Quick stats ---
print(f"\nProcessed - min: {proc.min():.4f}, max: {proc.max():.4f}, mean: {proc.mean():.4f}")
print(f"Raw TPM   - min: {raw.min():.4f}, max: {raw.max():.4f}, mean: {raw.mean():.4f}")

# Top discrepancies
diff = (proc - raw).abs()
top_diff = diff.nlargest(10)
print(f"\nTop 10 absolute differences:")
for gene, d in top_diff.items():
    print(f"  {gene:>15s}: proc={proc[gene]:.2f}  raw_tpm={raw[gene]:.2f}  diff={d:.2f}")
