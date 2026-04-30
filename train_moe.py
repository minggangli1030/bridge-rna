#!/usr/bin/env python3
# coding=utf-8

"""
MoE proof-of-concept training: 3 frozen 5k experts + small gating network.

Loads `human_5k`, `mouse_5k`, `mixed_5k` checkpoints, freezes them, then
trains a per-sample softmax gate over their predictions on the combined
12k-sample union. Single-GPU; no DDP. Designed to finish in 24–48h.

Success criterion: MoE val loss / OSDR pearson beats the best single expert
(mouse_5k @ 0.781 random_15pct).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from train_single import (
    ExpressionPerformer,
    InMemoryExpressionMLMDataset,
    load_selected_expression_rows,
    build_single_parquet_split,
    get_num_genes_from_single_parquet,
    format_duration,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================
# DEFAULTS — overridable via CLI
# ============================================================
DEFAULT_EXPERTS = {
    "human_5k": {
        "checkpoint": "./checkpoints/human_5k/best_model.pt",
        "expression_parquet": "./data/archs4/human_5k_merged/expression.parquet",
        "samples_json": "./data/archs4/human_5k/samples.json",
        "balanced_sampling": False,
    },
    "mouse_5k": {
        "checkpoint": "./checkpoints/mouse_5k/best_model.pt",
        "expression_parquet": "./data/archs4/mouse_5k_merged/expression.parquet",
        "samples_json": "./data/archs4/mouse_5k/samples.json",
        "balanced_sampling": False,
    },
    "mixed_5k": {
        "checkpoint": "./checkpoints/mixed_5k/best_model.pt",
        "expression_parquet": "./data/archs4/mixed_5k_merged/expression.parquet",
        "samples_json": "./data/archs4/mixed_5k/samples.json",
        "balanced_sampling": True,
    },
}

CONFIG = {
    "mask_ratio": 0.15,
    "mask_token": -10,
    "normalization": "log1p_tpm",
    "train_subset": 4000,
    "val_subset": 800,
    "batch_size": 4,
    "epochs": 30,
    "patience": 5,
    "early_stopping": True,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "gate_hidden": 256,
    "gate_dropout": 0.1,
    "use_amp": True,
    "seed": 42,
    "checkpoint_dir": "./checkpoints/moe_5k_v1",
    "log_every_frac": 4,  # log 4 times per epoch
}


# ============================================================
# GATING NETWORK
# ============================================================
class GatingNetwork(nn.Module):
    """Per-sample softmax over experts. Input: masked expression vector (B, G)."""

    def __init__(self, num_genes, num_experts, hidden_dim=256, dropout=0.1, mask_token=-10):
        super().__init__()
        self.mask_token = mask_token
        self.fc1 = nn.Linear(num_genes, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x_masked):
        # Replace mask token with 0 so it doesn't pollute the linear projection.
        x = torch.where(x_masked == self.mask_token, torch.zeros_like(x_masked), x_masked)
        h = self.act(self.fc1(x))
        h = self.dropout(h)
        logits = self.fc2(h)
        return F.softmax(logits, dim=-1)  # (B, num_experts)


# ============================================================
# EXPERT LOADING
# ============================================================
def load_expert(checkpoint_path, num_genes, device):
    """Reconstruct ExpressionPerformer from saved config and load weights. Frozen + eval mode."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    ckpt_num_genes = int(cfg.get("num_genes", num_genes))
    if ckpt_num_genes != num_genes:
        raise ValueError(
            f"Expert at {checkpoint_path} has num_genes={ckpt_num_genes}, "
            f"but combined data num_genes={num_genes}. Re-indexing not supported "
            f"in this proof-of-concept."
        )

    model = ExpressionPerformer(
        num_genes=num_genes,
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["num_heads"],
        n_layers=cfg["num_layers"],
        ffn_dim=cfg["ffn_dim"],
        ree_base=cfg["ree_base"],
        mask_token_id=cfg["mask_token"],
        feature_type=cfg["feature_type"],
        compute_type=cfg["compute_type"],
        gradient_checkpointing=False,
    )

    sd = ckpt["model_state_dict"]
    # Strip DDP "module." prefix if present.
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, cfg


# ============================================================
# DATA
# ============================================================
def verify_gene_alignment(parquet_paths):
    """All experts share the same gene_embedding indexing, so the combined data
    must use the same column order across all 3 parquets. Hard error if not."""
    ref = None
    for p in parquet_paths:
        cols = [
            c for c in pq.ParquetFile(str(p)).schema_arrow.names
            if c not in ("geo_accession", "__index_level_0__", "sample_id")
        ]
        if ref is None:
            ref = cols
        elif cols != ref:
            raise ValueError(
                f"Gene column order mismatch between parquets:\n  {parquet_paths[0]}\n  {p}\n"
                f"first diff at index {next((i for i, (a, b) in enumerate(zip(ref, cols)) if a != b), -1)}"
            )
    return ref


def build_combined_inmem_dataset(experts_cfg, cfg, seed, verbose=True):
    """Concatenate train+val splits across all 3 variants. Returns:
       - train_x (N_train, G), val_x (N_val, G) numpy arrays
       - train_variant (N_train,), val_variant (N_val,) int arrays (0/1/2)
    """
    parquet_paths = [Path(v["expression_parquet"]) for v in experts_cfg.values()]
    gene_cols = verify_gene_alignment(parquet_paths)
    num_genes = len(gene_cols)
    if verbose:
        print(f"[DATA] Gene alignment OK across {len(parquet_paths)} parquets ({num_genes} genes).", flush=True)

    train_chunks, val_chunks = [], []
    train_var, val_var = [], []
    for vidx, (name, vcfg) in enumerate(experts_cfg.items()):
        if verbose:
            print(f"\n[DATA] Variant {name}", flush=True)
        train_rows, val_rows = build_single_parquet_split(
            parquet_path=Path(vcfg["expression_parquet"]),
            samples_json_path=vcfg["samples_json"],
            train_subset=cfg["train_subset"],
            val_subset=cfg["val_subset"],
            balanced_sampling=vcfg.get("balanced_sampling", False),
            seed=seed,
            verbose=verbose,
        )
        t0 = time.time()
        train_x = load_selected_expression_rows(Path(vcfg["expression_parquet"]), train_rows)
        val_x = load_selected_expression_rows(Path(vcfg["expression_parquet"]), val_rows)
        if verbose:
            print(f"  Loaded train={train_x.shape} val={val_x.shape} in {time.time() - t0:.1f}s", flush=True)
        train_chunks.append(train_x)
        val_chunks.append(val_x)
        train_var.append(np.full(len(train_x), vidx, dtype=np.int64))
        val_var.append(np.full(len(val_x), vidx, dtype=np.int64))

    train_x = np.concatenate(train_chunks, axis=0)
    val_x = np.concatenate(val_chunks, axis=0)
    train_var = np.concatenate(train_var, axis=0)
    val_var = np.concatenate(val_var, axis=0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_x))
    train_x = train_x[perm]
    train_var = train_var[perm]

    if verbose:
        print(f"\n[DATA] Combined train: {train_x.shape}, val: {val_x.shape}", flush=True)
        gb = (train_x.nbytes + val_x.nbytes) / (1024 ** 3)
        print(f"[DATA] Memory footprint: {gb:.2f} GiB", flush=True)

    return train_x, val_x, train_var, val_var, num_genes


class CombinedMLMDataset(InMemoryExpressionMLMDataset):
    """Same as InMemoryExpressionMLMDataset but also returns the variant id per sample."""

    def __init__(self, x_data, variant_ids, normalization="log1p_tpm", mask_ratio=0.15, mask_token=-10):
        super().__init__(x_data, normalization=normalization, mask_ratio=mask_ratio, mask_token=mask_token)
        self.variant_ids = np.asarray(variant_ids, dtype=np.int64)

    def collate_batch(self, batch_record_indices):
        x_masked, x_true, mask_idx = super().collate_batch(batch_record_indices)
        rows = np.asarray(batch_record_indices, dtype=np.int64)
        var_ids = torch.from_numpy(self.variant_ids[rows])
        return x_masked, x_true, mask_idx, var_ids


# ============================================================
# TRAIN / EVAL LOOPS
# ============================================================
def masked_mse(pred, x_true, mask_idx):
    parts = []
    for i in range(pred.shape[0]):
        idxs = mask_idx[i]
        if len(idxs) > 0:
            parts.append(F.mse_loss(pred[i, idxs], x_true[i, idxs]))
    return torch.stack(parts).mean() if parts else torch.tensor(0.0, device=pred.device)


def expert_forward_stack(experts, x_masked):
    """Run each frozen expert under no_grad; return tensor (E, B, G)."""
    outs = []
    with torch.no_grad():
        for m in experts:
            outs.append(m(x_masked))
    return torch.stack(outs, dim=0)  # (E, B, G)


def combine_predictions(expert_preds, gate_weights):
    """expert_preds: (E, B, G). gate_weights: (B, E). Returns (B, G)."""
    # (B, E, 1) * (B, E, G) -> sum over E
    w = gate_weights.unsqueeze(-1)                  # (B, E, 1)
    p = expert_preds.permute(1, 0, 2)               # (B, E, G)
    return (w * p).sum(dim=1)                       # (B, G)


def evaluate(gate, experts, val_loader, device, use_amp, num_experts):
    gate.eval()
    val_loss_sum = 0.0
    val_batches = 0
    weight_sums = torch.zeros(num_experts, device=device)
    weight_count = 0
    # per-variant accounting
    per_variant_loss = torch.zeros(num_experts, device=device)
    per_variant_count = torch.zeros(num_experts, device=device)

    with torch.no_grad():
        for x_masked, x_true, mask_idx, var_ids in val_loader:
            x_masked = x_masked.to(device, non_blocking=True)
            x_true = x_true.to(device, non_blocking=True)
            var_ids = var_ids.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                expert_preds = expert_forward_stack(experts, x_masked)  # (E, B, G)
                gate_w = gate(x_masked)                                  # (B, E)
                pred = combine_predictions(expert_preds, gate_w)         # (B, G)
                loss = masked_mse(pred, x_true, mask_idx)

            val_loss_sum += loss.item()
            val_batches += 1
            weight_sums += gate_w.sum(dim=0)
            weight_count += gate_w.shape[0]

            # per-sample masked-MSE for per-variant breakdown
            for i in range(pred.shape[0]):
                idxs = mask_idx[i]
                if len(idxs) > 0:
                    sample_loss = F.mse_loss(pred[i, idxs], x_true[i, idxs])
                    v = int(var_ids[i].item())
                    per_variant_loss[v] += sample_loss
                    per_variant_count[v] += 1

    avg_weights = (weight_sums / max(1, weight_count)).cpu().tolist()
    avg_loss = val_loss_sum / max(1, val_batches)
    pv_loss = (per_variant_loss / per_variant_count.clamp(min=1)).cpu().tolist()
    pv_count = per_variant_count.cpu().tolist()
    return avg_loss, avg_weights, pv_loss, pv_count


def main():
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--human-ckpt", default=DEFAULT_EXPERTS["human_5k"]["checkpoint"])
    parser.add_argument("--mouse-ckpt", default=DEFAULT_EXPERTS["mouse_5k"]["checkpoint"])
    parser.add_argument("--mixed-ckpt", default=DEFAULT_EXPERTS["mixed_5k"]["checkpoint"])
    parser.add_argument("--checkpoint-dir", default=CONFIG["checkpoint_dir"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--mask-ratio", type=float, default=CONFIG["mask_ratio"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    CONFIG["checkpoint_dir"] = args.checkpoint_dir
    CONFIG["epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    CONFIG["learning_rate"] = args.lr
    CONFIG["mask_ratio"] = args.mask_ratio
    CONFIG["seed"] = args.seed
    CONFIG["use_amp"] = not args.no_amp

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda:0")

    print("\n" + "=" * 70)
    print("MoE Gating Network Training (3 frozen 5k experts)")
    print("=" * 70)
    print(f"[SETUP] Device: {device}")
    print(f"[SETUP] CONFIG: {json.dumps({k: v for k, v in CONFIG.items() if not callable(v)}, indent=2)}")

    expert_paths = {
        "human_5k": args.human_ckpt,
        "mouse_5k": args.mouse_ckpt,
        "mixed_5k": args.mixed_ckpt,
    }
    expert_names = list(expert_paths.keys())
    experts_cfg = {k: {**DEFAULT_EXPERTS[k], "checkpoint": expert_paths[k]} for k in expert_names}

    for name, vcfg in experts_cfg.items():
        if not Path(vcfg["checkpoint"]).exists():
            raise FileNotFoundError(f"Missing expert checkpoint: {name} -> {vcfg['checkpoint']}")
        if not Path(vcfg["expression_parquet"]).exists():
            raise FileNotFoundError(f"Missing parquet: {name} -> {vcfg['expression_parquet']}")

    parquet_paths = [Path(v["expression_parquet"]) for v in experts_cfg.values()]
    num_genes = get_num_genes_from_single_parquet(parquet_paths[0])
    assert num_genes > 10000, f"Unexpected num_genes={num_genes}"

    print("\n[EXPERTS] Loading frozen expert models...", flush=True)
    experts = []
    for name, vcfg in experts_cfg.items():
        m, _cfg = load_expert(vcfg["checkpoint"], num_genes=num_genes, device=device)
        params = sum(p.numel() for p in m.parameters())
        print(f"  + {name}: {params:,} params  (from {vcfg['checkpoint']})", flush=True)
        experts.append(m)

    print("\n[DATA] Building combined dataset...", flush=True)
    train_x, val_x, train_var, val_var, num_genes_data = build_combined_inmem_dataset(
        experts_cfg, CONFIG, seed=CONFIG["seed"], verbose=True
    )
    assert num_genes_data == num_genes, f"Gene count mismatch: experts={num_genes} data={num_genes_data}"

    train_ds = CombinedMLMDataset(
        train_x, train_var,
        normalization=CONFIG["normalization"],
        mask_ratio=CONFIG["mask_ratio"],
        mask_token=CONFIG["mask_token"],
    )
    val_ds = CombinedMLMDataset(
        val_x, val_var,
        normalization=CONFIG["normalization"],
        mask_ratio=CONFIG["mask_ratio"],
        mask_token=CONFIG["mask_token"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=train_ds.collate_batch,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=val_ds.collate_batch,
        num_workers=0,
        pin_memory=True,
    )
    print(f"[DATA] Train: {len(train_ds):,} samples, {len(train_loader)} batches", flush=True)
    print(f"[DATA] Val:   {len(val_ds):,} samples, {len(val_loader)} batches", flush=True)

    print("\n[MODEL] Building gating network...", flush=True)
    gate = GatingNetwork(
        num_genes=num_genes,
        num_experts=len(experts),
        hidden_dim=CONFIG["gate_hidden"],
        dropout=CONFIG["gate_dropout"],
        mask_token=CONFIG["mask_token"],
    ).to(device)
    gate_params = sum(p.numel() for p in gate.parameters())
    print(f"  + Gate params: {gate_params:,}", flush=True)

    optimizer = AdamW(gate.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=CONFIG["use_amp"])

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "bridge-rna"),
            group="moe_5k",
            name=os.environ.get("WANDB_RUN_NAME", "moe_5k_v1"),
            config={**CONFIG, "experts": expert_names, "expert_paths": expert_paths, "num_genes": num_genes},
        )

    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("[TRAIN] Starting training...")
    print("=" * 70 + "\n", flush=True)

    script_start = time.time()
    best_val = float("inf")
    patience = 0
    history = []

    for epoch in range(CONFIG["epochs"]):
        gate.train()
        epoch_start = time.time()
        running = 0.0
        nb = 0
        weight_sum = torch.zeros(len(experts), device=device)
        weight_count = 0
        log_every = max(1, len(train_loader) // CONFIG["log_every_frac"])

        for bi, (x_masked, x_true, mask_idx, _var_ids) in enumerate(train_loader):
            x_masked = x_masked.to(device, non_blocking=True)
            x_true = x_true.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=CONFIG["use_amp"]):
                expert_preds = expert_forward_stack(experts, x_masked)
                gate_w = gate(x_masked)
                pred = combine_predictions(expert_preds, gate_w)
                loss = masked_mse(pred, x_true, mask_idx)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            nb += 1
            weight_sum += gate_w.detach().sum(dim=0)
            weight_count += gate_w.shape[0]

            if (bi + 1) % log_every == 0:
                elapsed = time.time() - epoch_start
                spb = elapsed / nb
                eta_epoch = spb * (len(train_loader) - (bi + 1))
                done_total = epoch * len(train_loader) + (bi + 1)
                total_batches = CONFIG["epochs"] * len(train_loader)
                eta_run = (time.time() - script_start) / max(1, done_total) * (total_batches - done_total)
                print(
                    f"  Epoch {epoch+1}/{CONFIG['epochs']} | "
                    f"Batch {bi+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f} | Avg: {running/nb:.6f} | "
                    f"{spb:.3f}s/batch | "
                    f"Epoch ETA: {format_duration(eta_epoch)} | "
                    f"Run ETA: {format_duration(eta_run)}",
                    flush=True,
                )

        train_loss = running / max(1, nb)
        train_w = (weight_sum / max(1, weight_count)).cpu().tolist()

        val_loss, val_w, pv_loss, pv_count = evaluate(
            gate, experts, val_loader, device, CONFIG["use_amp"], len(experts)
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start

        print("\n  +==========================================+")
        print(f"  | Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"  | Train Loss: {train_loss:.6f}")
        print(f"  | Val Loss:   {val_loss:.6f}")
        print(f"  | Time: {epoch_time:.1f}s")
        print(f"  | Avg gate weights (train): " + ", ".join(f"{n}={w:.3f}" for n, w in zip(expert_names, train_w)))
        print(f"  | Avg gate weights (val):   " + ", ".join(f"{n}={w:.3f}" for n, w in zip(expert_names, val_w)))
        print(f"  | Per-variant val loss: " + ", ".join(
            f"{n}={l:.4f} (n={int(c)})" for n, l, c in zip(expert_names, pv_loss, pv_count)
        ))

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_w_{n}": w for n, w in zip(expert_names, val_w)},
            **{f"val_loss_{n}": l for n, l in zip(expert_names, pv_loss)},
        })

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
                **{f"train_w_{n}": w for n, w in zip(expert_names, train_w)},
                **{f"val_w_{n}": w for n, w in zip(expert_names, val_w)},
                **{f"val_loss_{n}": l for n, l in zip(expert_names, pv_loss)},
            })

        payload = {
            "gate_state_dict": gate.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": CONFIG,
            "expert_names": expert_names,
            "expert_paths": expert_paths,
            "num_genes": num_genes,
            "gate_arch": {
                "type": "GatingNetwork",
                "num_genes": num_genes,
                "num_experts": len(experts),
                "hidden_dim": CONFIG["gate_hidden"],
                "dropout": CONFIG["gate_dropout"],
                "mask_token": CONFIG["mask_token"],
            },
        }
        torch.save(payload, ckpt_dir / f"epoch_{epoch:02d}.pt")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(payload, ckpt_dir / "best_gate.pt")
            print(f"  | + New best! Saved best_gate.pt (val={best_val:.6f})")
        else:
            patience += 1
            print(f"  | - No improvement ({patience}/{CONFIG['patience']})")
            if CONFIG["early_stopping"] and patience >= CONFIG["patience"]:
                print("  | ! Early stopping!")
                print("  +==========================================+\n")
                break
        print("  +==========================================+\n", flush=True)

    pd.DataFrame(history).to_csv(ckpt_dir / "loss_history.csv", index=False)
    with open(ckpt_dir / "run_metadata.json", "w") as f:
        json.dump({
            "best_val_loss": best_val,
            "expert_names": expert_names,
            "expert_paths": expert_paths,
            "config": CONFIG,
            "num_genes": num_genes,
            "elapsed_seconds": time.time() - script_start,
        }, f, indent=2)

    print(f"\n[DONE] Best val loss: {best_val:.6f}")
    print(f"[DONE] Best gate saved to: {ckpt_dir / 'best_gate.pt'}")
    print(f"[DONE] Total time: {format_duration(time.time() - script_start)}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
