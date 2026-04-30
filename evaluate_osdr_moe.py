#!/usr/bin/env python3
# coding=utf-8
"""
Zero-shot OSDR evaluation for the MoE gate (`train_moe.py` output).

Loads `best_gate.pt` plus the 3 expert checkpoints it references, builds a
thin wrapper module with the same `forward(x) -> (B, G)` signature as
ExpressionPerformer, then reuses `evaluate_osdr.evaluate_model` to compute
the same metrics under the same masking strategies as single-expert eval.

Usage:
  python evaluate_osdr_moe.py \\
    --gate checkpoints/moe_5k_v1/best_gate.pt \\
    --osdr-parquet data/osdr/osdr_expression.parquet \\
    --output-dir results/osdr_eval_moe_5k_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import evaluate_osdr as eo
from train_moe import GatingNetwork


class MoEWrapper(nn.Module):
    """Thin module: 3 frozen experts + softmax gate → weighted-sum prediction."""

    def __init__(self, experts: list[nn.Module], gate: nn.Module):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        for m in self.experts:
            for p in m.parameters():
                p.requires_grad = False
        for p in self.gate.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = torch.stack([m(x) for m in self.experts], dim=0)  # (E, B, G)
        w = self.gate(x).unsqueeze(-1)                            # (B, E, 1)
        return (w * outs.permute(1, 0, 2)).sum(dim=1)             # (B, G)


def load_moe(gate_ckpt_path: Path, device: torch.device, expert_path_overrides: dict | None = None):
    """Load gate + 3 experts. Returns (wrapper, expert_train_gene_list, mask_token)."""
    gate_ckpt = torch.load(gate_ckpt_path, map_location="cpu", weights_only=False)
    arch = gate_ckpt["gate_arch"]
    expert_paths = dict(gate_ckpt["expert_paths"])
    expert_names = list(gate_ckpt["expert_names"])

    if expert_path_overrides:
        expert_paths.update(expert_path_overrides)

    print(f"[MOE] Gate from {gate_ckpt_path}")
    print(f"[MOE] Gate arch: {arch}")
    print(f"[MOE] Experts (in gate-order):")
    for n in expert_names:
        print(f"  {n}: {expert_paths[n]}")

    experts = []
    train_gene_lists = []
    mask_tokens = []
    for n in expert_names:
        m, cfg, gl = eo.load_checkpoint(Path(expert_paths[n]), device)
        for p in m.parameters():
            p.requires_grad = False
        experts.append(m)
        train_gene_lists.append(gl)
        mask_tokens.append(int(cfg.get("mask_token", -10)))

    # All experts must share the same gene order (the gate maps directly into expert input space).
    if any(gl != train_gene_lists[0] for gl in train_gene_lists if gl):
        raise ValueError("Experts have mismatched training gene orders — MoE cannot combine them.")
    if len(set(mask_tokens)) > 1:
        raise ValueError(f"Experts have mismatched mask tokens: {dict(zip(expert_names, mask_tokens))}")

    gate = GatingNetwork(
        num_genes=arch["num_genes"],
        num_experts=arch["num_experts"],
        hidden_dim=arch["hidden_dim"],
        dropout=arch["dropout"],
        mask_token=arch["mask_token"],
    )
    gate.load_state_dict(gate_ckpt["gate_state_dict"])
    gate.to(device)
    gate.eval()

    wrapper = MoEWrapper(experts, gate).to(device)
    wrapper.eval()
    return wrapper, train_gene_lists[0], int(arch["mask_token"]), expert_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, help="Path to best_gate.pt from train_moe.py")
    parser.add_argument("--output-dir", default="results/osdr_eval_moe")
    parser.add_argument("--osdr-parquet", default=None)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--osdr-raw-dir", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=eo.BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variant-name", default="moe", help="Label used in results CSV/JSON and W&B")
    parser.add_argument("--wandb", action="store_true")
    # Optional override: re-point experts at different checkpoint paths
    parser.add_argument("--human-ckpt", default=None)
    parser.add_argument("--mouse-ckpt", default=None)
    parser.add_argument("--mixed-ckpt", default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cpu":
        n_threads = int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads()))
        torch.set_num_threads(n_threads)
        print(f"[EVAL] Device: cpu ({n_threads} threads)", flush=True)
    else:
        print(f"[EVAL] Device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reference gene data + OSDR loading via evaluate_osdr's helpers
    print("[EVAL] Loading reference gene data...", flush=True)
    ortholog_map = eo.load_ortholog_map()
    canonical_genes = eo.load_canonical_genes()
    exon_lengths = eo.load_mouse_exon_lengths()
    print(f"  + {len(ortholog_map):,} ortholog pairs, {len(canonical_genes):,} canonical genes")

    if args.osdr_parquet:
        osdr_parquet = Path(args.osdr_parquet)
        if not osdr_parquet.exists():
            raise FileNotFoundError(f"OSDR parquet not found: {osdr_parquet}")
        print(f"[EVAL] Loading OSDR parquet: {osdr_parquet}", flush=True)
        osdr_df = pd.read_parquet(osdr_parquet)
    else:
        metadata_csv = Path(args.metadata_csv) if args.metadata_csv else eo.DEFAULT_METADATA_CSV
        raw_dir = Path(args.osdr_raw_dir) if args.osdr_raw_dir else None
        osdr_df = eo.preprocess_osdr(
            metadata_csv=metadata_csv,
            osdr_raw_dir=raw_dir,
            cache_dir=eo.DEFAULT_OSDR_CACHE,
            ortholog_map=ortholog_map,
            canonical_genes=canonical_genes,
            exon_lengths=exon_lengths,
            download=not args.no_download,
        )

    print(f"[EVAL] OSDR shape: {osdr_df.shape}", flush=True)
    gene_cols_present = [c for c in canonical_genes if c in osdr_df.columns]
    x_osdr_canonical = osdr_df[canonical_genes].values.astype("float32")
    osdr_gene_mask_canonical = np.array(
        [(g in set(gene_cols_present)) for g in canonical_genes], dtype=bool
    )

    # Load MoE
    expert_overrides = {}
    if args.human_ckpt:
        expert_overrides["human_5k"] = args.human_ckpt
    if args.mouse_ckpt:
        expert_overrides["mouse_5k"] = args.mouse_ckpt
    if args.mixed_ckpt:
        expert_overrides["mixed_5k"] = args.mixed_ckpt

    print("\n[EVAL] === Evaluating MoE wrapper ===", flush=True)
    t0 = time.time()
    wrapper, train_gene_list, mask_token, expert_names = load_moe(
        Path(args.gate), device, expert_path_overrides=expert_overrides
    )

    # Re-align OSDR to expert-training gene order if it differs from canonical
    if train_gene_list and train_gene_list != canonical_genes:
        print(f"  [INFO] Re-aligning OSDR to expert-training gene order "
              f"({len(train_gene_list)} genes vs {len(canonical_genes)} canonical)")
        x_aligned = (
            pd.DataFrame(x_osdr_canonical, columns=canonical_genes)
              .reindex(columns=train_gene_list, fill_value=0.0)
              .values.astype("float32")
        )
        gene_mask_aligned = np.array(
            [(g in set(gene_cols_present)) for g in train_gene_list], dtype=bool
        )
    else:
        x_aligned = x_osdr_canonical
        gene_mask_aligned = osdr_gene_mask_canonical

    total_params = sum(p.numel() for p in wrapper.parameters())
    print(f"  Wrapper params (experts + gate): {total_params:,}", flush=True)
    print(f"  Evaluating {x_aligned.shape[0]} samples, {x_aligned.shape[1]} genes ...", flush=True)

    metrics = eo.evaluate_model(
        model=wrapper,
        x_true=x_aligned,
        osdr_gene_mask=gene_mask_aligned,
        mask_ratios=eo.MASK_RATIOS,
        block_sizes=eo.BLOCK_SIZES,
        batch_size=args.batch_size,
        device=device,
        mask_token=float(mask_token),
    )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s", flush=True)

    # Pretty print
    print(f"\n  {'Masking':<20} {'Pearson':>10} {'Spearman':>10} {'MSE':>10} "
          f"{'BL_GeneMean':>12} {'BL_SampMean':>12}")
    print(f"  {'-'*76}")
    for strat, m in metrics.items():
        print(f"  {strat:<20} {m['pearson_mean']:>10.4f} {m['spearman_mean']:>10.4f} "
              f"{m['mse_mean']:>10.4f} {m['baseline_gene_mean_pearson']:>12.4f} "
              f"{m['baseline_sample_mean_pearson']:>12.4f}")

    rows = [
        {"variant": args.variant_name, "checkpoint": str(args.gate),
         "masking_strategy": strat, "experts": ",".join(expert_names), **m}
        for strat, m in metrics.items()
    ]
    pd.DataFrame(rows).to_csv(output_dir / "osdr_eval_moe_results.csv", index=False)
    with open(output_dir / "osdr_eval_moe_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[EVAL] Results saved to {output_dir}/")

    if args.wandb:
        try:
            import wandb as wb
            run = wb.init(
                project=os.environ.get("WANDB_PROJECT", "bridge-rna"),
                name=f"osdr_eval_{args.variant_name}",
                group="osdr_benchmark",
                config={"gate": str(args.gate), "experts": expert_names,
                        "n_samples": x_aligned.shape[0]},
            )
            for strat, m in metrics.items():
                wb.log({f"{args.variant_name}/{strat}/{k}": v for k, v in m.items()})
            run.log({"summary_table": wb.Table(dataframe=pd.DataFrame(rows))})
            run.finish()
        except Exception as e:
            print(f"[WARN] W&B logging failed: {e}", flush=True)


if __name__ == "__main__":
    main()
