#!/usr/bin/env python3
"""
Convert Karen Meng's preprocessed OSDR data to bridge-rna canonical gene space.

Input:  /global/scratch/users/kmeng/osdr_data/osdr_processed_all.csv
        /global/scratch/users/kmeng/osdr_data/spaceflight_labels.csv
Output: data/osdr/osdr_expression.parquet  (picked up by evaluate_osdr.py)

The OSDR CSV uses ENSMUSG IDs; this script maps them to human gene symbols
via the ortholog table and aligns to the bridge-rna canonical gene list.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OSDR_CSV     = Path("/global/scratch/users/kmeng/osdr_data/osdr_processed_all.csv")
LABELS_CSV   = Path("/global/scratch/users/kmeng/osdr_data/spaceflight_labels.csv")
ORTHOLOG_CSV = Path(__file__).parent / "data" / "osdr" / "human_mouse_orthologs.csv"
CANONICAL_TXT = Path(__file__).parent / "data" / "ensembl" / "protein_coding_ortholog_genes.txt"
OUT_PARQUET  = Path(__file__).parent / "data" / "osdr" / "osdr_expression.parquet"

def main():
    print("Loading OSDR expression data...", flush=True)
    expr = pd.read_csv(OSDR_CSV, index_col=0)
    print(f"  Loaded: {expr.shape}  (samples x ENSMUSG genes)", flush=True)

    print("Loading spaceflight labels...", flush=True)
    labels = pd.read_csv(LABELS_CSV, index_col=0)
    # index is integer, sample name in 'id.sample name' col
    label_map = dict(zip(labels["id.sample name"], labels["spaceflight"]))
    print(f"  {sum(v==1 for v in label_map.values())} spaceflight, "
          f"{sum(v==0 for v in label_map.values())} control", flush=True)

    print("Building ENSMUSG → human gene symbol map...", flush=True)
    orth = pd.read_csv(ORTHOLOG_CSV)
    orth = orth[orth["Mouse homology type"] == "ortholog_one2one"]
    ensmusg_to_human = dict(zip(orth["Mouse gene stable ID"].str.strip(),
                                orth["Gene name"].str.strip()))
    print(f"  {len(ensmusg_to_human):,} one-to-one ENSMUSG → human pairs", flush=True)

    print("Loading canonical gene list...", flush=True)
    canonical_genes = pd.read_csv(CANONICAL_TXT, header=None)[0].str.strip().tolist()
    canonical_set = set(canonical_genes)
    print(f"  {len(canonical_genes):,} canonical genes", flush=True)

    print("Mapping columns ENSMUSG → human symbols...", flush=True)
    new_cols = [ensmusg_to_human.get(c, "") for c in expr.columns]
    expr.columns = new_cols
    expr = expr.loc[:, expr.columns != ""]          # drop unmapped
    expr = expr.loc[:, ~expr.columns.duplicated(keep="first")]  # drop duplicate human symbols

    matched = sum(1 for c in expr.columns if c in canonical_set)
    print(f"  {matched:,}/{len(canonical_genes):,} canonical genes present in OSDR", flush=True)

    print("Aligning to canonical gene order (filling missing with 0)...", flush=True)
    aligned = expr.reindex(columns=canonical_genes, fill_value=0.0).astype("float32")

    # Attach spaceflight label
    aligned["spaceflight"] = aligned.index.map(lambda s: label_map.get(s, np.nan))
    n_labeled   = aligned["spaceflight"].notna().sum()
    n_flight    = (aligned["spaceflight"] == 1).sum()
    n_control   = (aligned["spaceflight"] == 0).sum()
    n_unlabeled = aligned["spaceflight"].isna().sum()
    print(f"  Spaceflight labels: {n_flight} flight, {n_control} control, "
          f"{n_unlabeled} unlabeled", flush=True)

    aligned.index.name = "sample_id"

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(OUT_PARQUET)
    print(f"\nSaved: {OUT_PARQUET}  shape={aligned.shape}", flush=True)
    print("Done. Now run evaluate_osdr.py with --osdr-parquet data/osdr/osdr_expression.parquet")


if __name__ == "__main__":
    main()
