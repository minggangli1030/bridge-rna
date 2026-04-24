#!/usr/bin/env python3
"""
Diagnostic: is there a train/eval alignment bug that explains why models
lose to the gene_mean baseline on OSDR?

Hypotheses tested:
  [A] Gene order in training parquet differs from canonical_genes used at eval.
  [B] mask_token / normalization mismatch between training and eval.
  [C] Model predictions are constant or out-of-range (training collapse).
  [D] Random mask lands on zero-filled (OSDR-missing) positions, so the
      "true" value is a fill-zero with no signal to learn.
  [E] Model can't even recover expression at UNmasked positions (i.e. it
      produces garbage regardless of masking).

Usage:
  python check_alignment.py \\
    --checkpoint checkpoints/human_5k/best_model.pt \\
    --osdr-parquet data/osdr/osdr_expression.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_osdr import load_canonical_genes, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--osdr-parquet", required=True)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--mask-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"=== ALIGNMENT DIAGNOSTIC ===")
    print(f"checkpoint:   {args.checkpoint}")
    print(f"osdr parquet: {args.osdr_parquet}")
    print(f"device:       {device}\n")

    canonical_genes = load_canonical_genes()
    model, cfg, train_gene_list = load_checkpoint(Path(args.checkpoint), device)
    mask_token = float(cfg.get("mask_token", -10.0))

    # ── [1] Checkpoint config ──────────────────────────────────────────────
    print("[1] CHECKPOINT CONFIG")
    for k in ("num_genes", "hidden_dim", "num_layers", "num_heads",
              "mask_token", "normalization", "expression_parquet"):
        print(f"  {k:<22} {cfg.get(k)}")
    print(f"  {'param count':<22} {sum(x.numel() for x in model.parameters()):,}")
    print()

    # ── [2] Gene-order alignment ───────────────────────────────────────────
    print("[2] GENE ORDER (train parquet vs canonical_genes)")
    print(f"  canonical_genes : {len(canonical_genes)}")
    print(f"  train_gene_list : {len(train_gene_list) if train_gene_list else 'UNAVAILABLE (parquet path missing)'}")
    if train_gene_list:
        if train_gene_list == canonical_genes:
            print("  OK: orders match exactly")
        else:
            n = min(len(train_gene_list), len(canonical_genes))
            nmatch = sum(a == b for a, b in zip(train_gene_list[:n], canonical_genes[:n]))
            overlap = set(train_gene_list) & set(canonical_genes)
            print(f"  MISMATCH: {nmatch}/{n} positions aligned")
            print(f"  set overlap: {len(overlap)} / {len(set(train_gene_list) | set(canonical_genes))}")
            print(f"  first 10 train : {train_gene_list[:10]}")
            print(f"  first 10 canon : {canonical_genes[:10]}")
    print()

    # ── [3] OSDR input distribution ────────────────────────────────────────
    osdr_df = pd.read_parquet(args.osdr_parquet)
    osdr_cols = set(osdr_df.columns.tolist())
    x_osdr = osdr_df[canonical_genes].values.astype("float32")
    present_mask = np.array(
        [(g in osdr_cols) for g in canonical_genes], dtype=bool
    )
    nonzero_frac = (x_osdr != 0).mean(axis=1)

    print("[3] OSDR INPUT")
    print(f"  df shape                : {osdr_df.shape}")
    print(f"  canonical cols present  : {present_mask.sum()}/{len(canonical_genes)}")
    print(f"  x range                 : [{x_osdr.min():.3f}, {x_osdr.max():.3f}]")
    print(f"  x mean / std            : {x_osdr.mean():.3f} / {x_osdr.std():.3f}")
    print(f"  per-sample nonzero frac : mean={nonzero_frac.mean():.3f} min={nonzero_frac.min():.3f}")
    print("  (log1p(TPM) → expected range ~0-10, mean ~2-5, nonzero frac 0.5-0.9)")
    print()

    # ── [4] Prediction spot-check on masked positions ──────────────────────
    rng = np.random.default_rng(args.seed)
    n, G = args.n_samples, x_osdr.shape[1]
    batch = x_osdr[:n]
    num_mask = int(G * args.mask_ratio)
    mask_idx = np.stack([rng.choice(G, num_mask, replace=False) for _ in range(n)])

    x_masked = batch.copy()
    for i in range(n):
        x_masked[i, mask_idx[i]] = mask_token

    with torch.no_grad():
        pred = model(torch.from_numpy(x_masked).to(device)).cpu().numpy()

    print(f"[4] PREDICTION SPOT-CHECK ({n} samples, {args.mask_ratio:.0%} random mask)")
    print(f"  pred full range : [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"  pred mean/std   : {pred.mean():.3f} / {pred.std():.3f}")
    print(f"  true full range : [{batch.min():.3f}, {batch.max():.3f}]  mean/std {batch.mean():.3f}/{batch.std():.3f}")
    print()
    for i in range(n):
        p = pred[i, mask_idx[i]]
        t = batch[i, mask_idx[i]]
        r = float("nan") if np.std(p) < 1e-9 else float(np.corrcoef(p, t)[0, 1])
        flag = "  <-- pred is constant" if np.std(p) < 1e-9 else ""
        print(f"  sample {i}: r={r:+.3f}  "
              f"pred[{p.min():+.2f},{p.max():+.2f}] mean={p.mean():+.2f} std={p.std():.2f}  "
              f"true[{t.min():.2f},{t.max():.2f}] mean={t.mean():.2f}{flag}")
    print()

    # ── [5] Does the model just predict gene_mean? ─────────────────────────
    gene_mean = batch.mean(axis=0)
    sims = []
    for i in range(n):
        p = pred[i, mask_idx[i]]
        gm = gene_mean[mask_idx[i]]
        if np.std(p) > 1e-9 and np.std(gm) > 1e-9:
            sims.append(float(np.corrcoef(p, gm)[0, 1]))
    print("[5] IS THE MODEL COLLAPSING TO GENE-MEAN?")
    if sims:
        print(f"  mean corr(pred, gene_mean) over masked = {np.mean(sims):+.3f}")
        print("  (>0.9 → predictions are just the per-gene average across OSDR)")
    print()

    # ── [6] Identity-recovery at UNmasked positions ────────────────────────
    print("[6] IDENTITY RECOVERY AT UNMASKED POSITIONS")
    print("  (at unmasked positions the model sees the true value as input; "
          "good models still output ~= input. If r is low here, the model "
          "is broken independent of masking.)")
    for i in range(n):
        unmask = np.array(sorted(set(range(G)) - set(mask_idx[i])))[:1000]
        p, t = pred[i, unmask], batch[i, unmask]
        r = float("nan") if np.std(p) < 1e-9 else float(np.corrcoef(p, t)[0, 1])
        print(f"  sample {i}: r={r:+.3f}")
    print()

    # ── [7] Fraction of masked positions on zero-filled (OSDR-missing) genes ─
    print("[7] MASK LANDING ON ZERO-FILLED (OSDR-MISSING) GENES")
    missing_mask = ~present_mask
    frac_on_missing = np.array(
        [missing_mask[mask_idx[i]].mean() for i in range(n)]
    )
    print(f"  mean fraction of mask positions on OSDR-missing genes: "
          f"{frac_on_missing.mean():.3f}")
    print("  (if high, many 'true' values at masked positions are just 0 fill "
          "— a significant drag on per-sample pearson.)")
    print()

    # ── [8] Per-sample pearson split: present-only vs. include-missing ─────
    print("[8] PER-SAMPLE PEARSON, PRESENT-ONLY vs. INCLUDE-MISSING")
    r_all, r_present = [], []
    for i in range(n):
        idx_all = mask_idx[i]
        idx_pres = idx_all[present_mask[idx_all]]
        if len(idx_pres) < 3:
            continue
        p_all, t_all = pred[i, idx_all], batch[i, idx_all]
        p_pr, t_pr = pred[i, idx_pres], batch[i, idx_pres]
        if np.std(p_all) > 1e-9:
            r_all.append(float(np.corrcoef(p_all, t_all)[0, 1]))
        if np.std(p_pr) > 1e-9:
            r_present.append(float(np.corrcoef(p_pr, t_pr)[0, 1]))
    if r_all and r_present:
        print(f"  include-missing pearson mean : {np.mean(r_all):+.3f}")
        print(f"  present-only pearson mean    : {np.mean(r_present):+.3f}")
        print(f"  delta                        : {np.mean(r_present) - np.mean(r_all):+.3f}")
        print("  (large positive delta → fix eval to sample masks from present genes only.)")


if __name__ == "__main__":
    main()
