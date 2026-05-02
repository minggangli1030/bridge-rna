#!/usr/bin/env python3
"""Build a mixed mouse(OSDR) + human(TCGA) eval parquet.

Both inputs are expected to be canonical-space log1p TPM (OSDR from the
training preprocessing pipeline; TCGA from prep_tcga.py). This script just
samples N from each, tags with `species`, and concatenates. The output is
fed to `analyze_moe_headroom.py --osdr-parquet ...` (it doesn't actually
have to be OSDR — any canonical-space sample-major parquet works).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osdr-parquet", required=True,
                    help="OSDR (mouse) parquet — canonical-space log1p TPM.")
    ap.add_argument("--tcga-parquet", required=True,
                    help="TCGA parquet from prep_tcga.py.")
    ap.add_argument("--n-each", type=int, default=500,
                    help="Samples to draw from each species.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    osdr = pd.read_parquet(args.osdr_parquet)
    tcga = pd.read_parquet(args.tcga_parquet)
    print(f"[load] OSDR: {len(osdr)} samples × {osdr.shape[1]} cols")
    print(f"[load] TCGA: {len(tcga)} samples × {tcga.shape[1]} cols")

    n_o = min(args.n_each, len(osdr))
    n_t = min(args.n_each, len(tcga))
    osdr_idx = rng.choice(len(osdr), n_o, replace=False)
    tcga_idx = rng.choice(len(tcga), n_t, replace=False)

    mouse = osdr.iloc[osdr_idx].reset_index(drop=True)
    if "species" not in mouse.columns:
        mouse["species"] = "mouse"
    human = tcga.iloc[tcga_idx].reset_index(drop=True)
    if "species" not in human.columns:
        human["species"] = "human"

    # Defensive: align columns by union (both should be canonical-aligned
    # already, but cover the case where one has an extra metadata col).
    cols = list(dict.fromkeys(list(mouse.columns) + list(human.columns)))
    cols = [c for c in cols if c != "species"] + ["species"]
    mouse = mouse.reindex(columns=cols, fill_value=0.0)
    human = human.reindex(columns=cols, fill_value=0.0)

    mixed = pd.concat([mouse, human], ignore_index=True)
    print(f"[build] mixed: {len(mixed)} samples × {mixed.shape[1]-1} gene cols + species")
    print(f"[build] species:")
    for s, c in mixed["species"].value_counts().items():
        print(f"        {s}: {c}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_parquet(out, compression="zstd")
    print(f"[save] {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
