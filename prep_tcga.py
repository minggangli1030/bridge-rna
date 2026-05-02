#!/usr/bin/env python3
"""Convert a TCGA TPM matrix to a canonical-space log1p parquet.

Input:  a parquet whose columns are HGNC gene symbols + (optionally)
        sample_id / __index_level_0__, with values already in TPM space.
Output: a sample-major parquet with columns = canonical_genes (the experts'
        protein_coding_ortholog vocabulary) + a `species='human'` column,
        values = log1p(TPM), missing canonical genes filled with 0.

Sized to be cheap and rerunnable; runs locally in seconds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_osdr import load_canonical_genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcga-tpm", default="tcga_tpm_unstranded_matrix.parquet",
                    help="Input TCGA TPM parquet (gene-symbol cols).")
    ap.add_argument("--output", default="data/tcga/tcga_expression.parquet")
    args = ap.parse_args()

    canonical = load_canonical_genes()
    print(f"[setup] canonical genes: {len(canonical)}")

    df = pd.read_parquet(args.tcga_tpm)
    df = df.drop(columns=[c for c in ("__index_level_0__",) if c in df.columns])
    sample_id = df.pop("sample_id") if "sample_id" in df.columns else None
    print(f"[load] TCGA: {len(df)} samples × {len(df.columns)} gene cols")

    sums = df.sum(axis=1)
    print(f"[sanity] per-sample sum (TPM ~1e6 expected): "
          f"mean={sums.mean():.0f}  min={sums.min():.0f}  max={sums.max():.0f}")

    overlap = sum(1 for g in canonical if g in df.columns)
    missing = len(canonical) - overlap
    print(f"[align] canonical ∩ TCGA = {overlap}/{len(canonical)}  "
          f"(missing → fill 0: {missing})")

    df = np.log1p(df.astype("float32"))
    df = df.reindex(columns=canonical, fill_value=0.0).astype("float32")

    if sample_id is not None:
        df.insert(0, "sample_id", sample_id.values)
    df["species"] = "human"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="zstd")
    print(f"[save] {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"[save] shape: {df.shape}")


if __name__ == "__main__":
    main()
