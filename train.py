# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ExpressionBERT: Masked gene expression prediction using Transformer.

Adapted from Google's SLiMPerformer for continuous gene expression data.
Architecture:
  - Gene identity embedding (learned, like BERT token IDs)
  - Rotary Expression Embedding (REE) for value-based positional encoding
  - N Transformer layers (multi-head attention + FFN + LayerNorm)
  - Output projection for per-gene expression reconstruction

Training objective: MLM-style masking (mask 15% of genes, predict their expression)

Usage:
  torchrun --nproc_per_node=2 train.py
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.distributed as dist
import pandas as pd

from slim_performer_model import SLiMPerformerLayer

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'hidden_dim': 256,
    'ffn_dim': 1024,
    'num_heads': 8,
    'num_layers': 2,
    'ree_base': 100.0,
    'feature_type': 'sqr',       # Linear attention kernel: 'relu', 'elu+1', 'sqr', 'favor+'
    'compute_type': 'iter',      # Prefix sum method: 'iter', 'ps', 'parallel_ps'
    'normalization': 'tpm',      # 'tpm' or 'log1p_tpm' applied before REE/model input
    'mask_ratio': 0.15,
    'mask_token': -10,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'batch_size': 16,
    'epochs': 5,
    'early_stopping': True,
    'patience': 5,
    'seed': 42,
    # Data subset sizes (set to None for full data)
    'train_subset': 10000,
    'val_subset': 2000,
    'balanced_sampling': True,
    'data_dir': './data/archs4/train_orthologs',
    'checkpoint_dir': './checkpoints_performer',
}


# ============================================================
# ROTARY EXPRESSION EMBEDDING (REE)
# ============================================================
class RotaryExpressionEmbedding(nn.Module):
    """
    Rotary Expression Embedding (REE): Converts continuous gene expression
    values into sinusoidal rotation features.

    Modulates rotary positional encodings using expression magnitude.
    Includes masking support for special tokens (e.g., masked expression = -10).
    Original base=100 (from Google SLiMPerformer research).
    """

    def __init__(self, dim, base=100.0, mask_token_id=-10):
        super().__init__()
        self.dim = dim
        self.mask_token_id = mask_token_id

        # inv_freq for sinusoidal encoding
        # base=100 (from original code) vs 10000 (standard Transformer)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_genes] expression values

        Returns:
            [batch_size, num_genes, dim] sinusoidal encodings
        """
        # Identify masked positions
        x_mask_idx = (x == self.mask_token_id).nonzero(as_tuple=False)

        # Multiply expression values by frequencies: [B, G] x [D/2] → [B, G, D/2]
        freqs = torch.einsum("bi,j->bij", x, self.inv_freq)

        # Apply sin and cos, then concatenate: [B, G, D/2] → [B, G, D]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)

        # Mask out special token positions (set to 0)
        if len(x_mask_idx) > 0:
            emb[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = 0

        return emb


# ============================================================
# EXPRESSION PERFORMER MODEL
# ============================================================
class ExpressionPerformer(nn.Module):
    """
    ExpressionBERT: Transformer for continuous gene expression data.
    Uses SLiMPerformer's linear attention (O(n) memory) from Google Research.

    Input:  [batch, num_genes] expression values (with masked positions = -10)
    Output: [batch, num_genes] predicted expression values

    Embeddings (summed, like BERT):
      1. Gene identity embedding — learned per-gene vector (like BERT token IDs)
      2. REE — sinusoidal encoding driven by expression magnitude
    """

    def __init__(self, num_genes, hidden_dim=256, n_heads=8, n_layers=4,
                 ffn_dim=1024, ree_base=100.0, mask_token_id=-10,
                 feature_type='sqr', compute_type='iter'):
        super().__init__()
        self.num_genes = num_genes
        self._hidden_dim = hidden_dim

        # Gene identity embedding (like BERT's token embedding)
        self.gene_embedding = nn.Embedding(num_genes, hidden_dim)

        # Rotary Expression Embedding
        self.ree = RotaryExpressionEmbedding(hidden_dim, base=ree_base,
                                              mask_token_id=mask_token_id)

        # SLiMPerformer layers (linear O(n) attention via prefix sums)
        self.layers = nn.ModuleList([
            SLiMPerformerLayer(hidden_dim, ffn_dim, n_heads,
                               feature_type, compute_type, on_gptln=True)
            for _ in range(n_layers)
        ])

        # Output: predict single expression value per gene
        self.output_map = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: [batch, num_genes] expression values
        Returns:
            [batch, num_genes] predicted expression
        """
        B, G = x.shape
        device = x.device

        # Gene identity embeddings: [G, hidden_dim] → broadcast to [B, G, hidden_dim]
        gene_ids = torch.arange(G, device=device)
        gene_emb = self.gene_embedding(gene_ids)

        # REE from expression values: [B, G, hidden_dim]
        ree_emb = self.ree(x)

        # Sum embeddings (like BERT: token + position)
        h = gene_emb.unsqueeze(0) + ree_emb

        # Pass through SLiMPerformer layers (linear attention)
        for layer in self.layers:
            rfs = layer.attention.sample_rfs(device)
            h = layer.full_forward(h, rfs)

        # Project to scalar per gene
        out = self.output_map(h).squeeze(-1)  # [B, G]

        return out


# ============================================================
# DATASET
# ============================================================
class ExpressionMLMDataset(Dataset):
    """Expression dataset with MLM-style masking."""

    def __init__(self, expr_array, mask_ratio=0.15, mask_token=-10):
        """
        Args:
            expr_array: [samples, genes] numpy array
            mask_ratio: fraction of genes to mask per sample
            mask_token: value for masked positions
        """
        self.X = expr_array.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        num_genes = x.shape[0]

        num_mask = max(1, int(num_genes * self.mask_ratio))
        mask_indices = np.random.choice(num_genes, num_mask, replace=False)

        x_masked = x.copy()
        x_masked[mask_indices] = self.mask_token

        return (
            torch.tensor(x_masked, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(mask_indices, dtype=torch.long),
        )


# ============================================================
# DATA LOADING
# ============================================================
def load_parquet_data(path):
    """Load parquet file and transpose to [samples, genes]."""
    df = pd.read_parquet(path)
    data = df.values.T.astype(np.float32)  # [samples, genes]
    return data


def _split_bounds(n: int) -> tuple[int, int]:
    """Return split boundaries for 80/10/10 train/val/test."""
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return n_train, n_train + n_val


def _load_unified_split_arrays(data_dir, split, species_list, seed=42, verbose=True):
    """Load split arrays from unified expression.parquet + metadata.csv layout.

    Expected unified files:
      - data_dir / expression.parquet (genes x samples)
      - data_dir / metadata.csv       (geo_accession, species)
    """
    expr_path = data_dir / 'expression.parquet'
    meta_path = data_dir / 'metadata.csv'
    if not expr_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Unified data files not found under {data_dir} "
            f"(expected expression.parquet and metadata.csv)"
        )

    if verbose:
        print(f"  ↳ Using unified dataset fallback from {expr_path}")

    # [samples, genes] with sample IDs as index
    expr_df = pd.read_parquet(expr_path).T
    meta_df = pd.read_csv(meta_path)
    if 'geo_accession' not in meta_df.columns or 'species' not in meta_df.columns:
        raise ValueError("metadata.csv must contain columns: geo_accession, species")

    arrays = {}
    raw_counts = {}

    for sp in species_list:
        sp_ids = meta_df.loc[meta_df['species'] == sp, 'geo_accession'].tolist()
        sp_ids = [sid for sid in sp_ids if sid in expr_df.index]
        if not sp_ids:
            if verbose:
                print(f"  ✗ {sp} {split}: no samples in unified metadata")
            continue

        rng = np.random.default_rng(seed + (abs(hash(sp)) % 10000))
        perm = rng.permutation(len(sp_ids))
        sp_ids = [sp_ids[i] for i in perm]

        n_train, n_val_end = _split_bounds(len(sp_ids))
        if split == 'train':
            ids = sp_ids[:n_train]
        elif split == 'val':
            ids = sp_ids[n_train:n_val_end]
        elif split == 'test':
            ids = sp_ids[n_val_end:]
        else:
            raise ValueError(f"Unknown split: {split!r}")

        arr = expr_df.loc[ids].values.astype(np.float32)
        arrays[sp] = arr
        raw_counts[sp] = int(arr.shape[0])
        if verbose:
            print(f"  ✓ {sp} {split} (unified): {arr.shape}")

    return arrays, raw_counts


def _sample_rows(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n >= len(arr):
        return arr
    idx = rng.choice(len(arr), size=n, replace=False)
    return arr[idx]


def apply_input_normalization(x: np.ndarray, normalization: str) -> np.ndarray:
    """Apply input normalization mode expected by the model."""
    if normalization == 'tpm':
        return x.astype(np.float32, copy=False)
    if normalization == 'log1p_tpm':
        # Clamp negatives to 0 before log1p for numerical stability.
        return np.log1p(np.maximum(x, 0.0)).astype(np.float32, copy=False)
    raise ValueError(f"Unknown normalization: {normalization!r}")


def format_float_for_tag(v: float) -> str:
    s = f"{v:.2e}" if (abs(v) < 1e-3 or abs(v) >= 1e3) else f"{v:.6f}"
    return s.replace('.', 'p').replace('+', '').replace('-', 'm')


def build_run_tag(cfg: dict) -> str:
    return (
        f"norm-{cfg['normalization']}"
        f"_lr-{format_float_for_tag(cfg['learning_rate'])}"
        f"_wd-{format_float_for_tag(cfg['weight_decay'])}"
        f"_mask-{format_float_for_tag(cfg['mask_ratio'])}"
        f"_ree-{format_float_for_tag(cfg['ree_base'])}"
    )


def load_split(
    data_dir,
    split,
    species_list,
    subset=None,
    balanced_sampling=True,
    seed=42,
    verbose=True,
):
    """Load split data and optionally subset with 50/50 species balancing.

    Returns:
        data: [samples, genes]
        used_counts: samples used per species after subsetting
        raw_counts: samples available per species before subsetting
    """
    arrays = {}
    raw_counts = {}
    for species in species_list:
        path = data_dir / split / f'expression_{split}_{species}.parquet'
        if path.exists():
            arr = load_parquet_data(path)
            if verbose:
                print(f"  ✓ {species} {split}: {arr.shape}")
            arrays[species] = arr
            raw_counts[species] = int(arr.shape[0])
        elif verbose:
            print(f"  ✗ {species} {split}: not found")

    # Fallback: unified layout (expression.parquet + metadata.csv)
    if not arrays:
        arrays, raw_counts = _load_unified_split_arrays(
            data_dir,
            split,
            species_list,
            seed=seed,
            verbose=verbose,
        )

    if not arrays:
        raise FileNotFoundError(f"No data found for split={split}")

    rng = np.random.default_rng(seed)

    if subset is not None:
        if balanced_sampling:
            # 50/50 species sampling for sweeps to avoid species dominance.
            per_species_target = max(1, subset // max(1, len(arrays)))
            max_balanced = min(arr.shape[0] for arr in arrays.values())
            per_species = min(per_species_target, max_balanced)
            arrays = {
                sp: _sample_rows(arr, per_species, rng)
                for sp, arr in arrays.items()
            }
            if verbose:
                print(
                    f"  ↳ Balanced subset enabled: {per_species} per species "
                    f"({per_species * len(arrays)} total)"
                )
        else:
            concat_all = np.vstack(list(arrays.values()))
            concat_all = _sample_rows(concat_all, subset, rng)
            used_counts = {sp: None for sp in arrays.keys()}
            return concat_all, used_counts, raw_counts

    used_counts = {sp: int(arr.shape[0]) for sp, arr in arrays.items()}
    return np.vstack(list(arrays.values())), used_counts, raw_counts


# ============================================================
# TRAINING (DDP)
# ============================================================
def main():
    script_start = time.time()

    # Initialize DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    is_main = rank == 0

    if is_main:
        print("\n" + "=" * 70)
        print(f"ExpressionPerformer Training — DDP ({world_size} GPUs)")
        print("=" * 70)
        print(f"\n[SETUP] Rank: {rank}, Device: {device}")

    # ─────────────────────────────────────────────────────────
    # WANDB (init early so sweep can override CONFIG)
    # ─────────────────────────────────────────────────────────
    if is_main and HAS_WANDB:
        wandb.init(
            project="expression-performer",
            config=CONFIG,
        )
        # When running a sweep, wandb.config overrides CONFIG values
        for key in CONFIG:
            if key in wandb.config:
                CONFIG[key] = wandb.config[key]

    # Broadcast CONFIG from rank 0 so all ranks use the same hyperparams
    config_list = [CONFIG if is_main else None]
    dist.broadcast_object_list(config_list, src=0)
    CONFIG.update(config_list[0])

    # ─────────────────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────────────────
    data_dir = Path(CONFIG['data_dir'])
    species = ['human', 'mouse']

    if is_main:
        print("\n[DATA] Loading training data...")
    t0 = time.time()
    X_train, train_used_counts, train_raw_counts = load_split(
        data_dir,
        'train',
        species,
        subset=CONFIG['train_subset'],
        balanced_sampling=CONFIG['balanced_sampling'],
        seed=CONFIG['seed'],
        verbose=is_main,
    )
    num_samples, num_genes = X_train.shape
    if is_main:
        print(f"  ✓ Combined: {X_train.shape}, Time: {time.time()-t0:.1f}s")
        print(f"  ↳ train used counts: {train_used_counts}")
        print(f"  ↳ train raw counts:  {train_raw_counts}")

    if is_main:
        print("\n[DATA] Loading validation data...")
    t0 = time.time()
    X_val, val_used_counts, val_raw_counts = load_split(
        data_dir,
        'val',
        species,
        subset=CONFIG['val_subset'],
        balanced_sampling=CONFIG['balanced_sampling'],
        seed=CONFIG['seed'] + 1,
        verbose=is_main,
    )
    if is_main:
        print(f"  ✓ Combined: {X_val.shape}, Time: {time.time()-t0:.1f}s")
        print(f"  ↳ val used counts: {val_used_counts}")
        print(f"  ↳ val raw counts:  {val_raw_counts}")

    # Apply selected normalization before sending values into REE/model.
    X_train = apply_input_normalization(X_train, CONFIG['normalization'])
    X_val = apply_input_normalization(X_val, CONFIG['normalization'])
    if is_main:
        print(
            f"[DATA] Input normalization='{CONFIG['normalization']}' | "
            f"train range=({X_train.min():.4f}, {X_train.max():.4f}) | "
            f"val range=({X_val.min():.4f}, {X_val.max():.4f})"
        )

    # Sanity checks
    if is_main:
        print(f"\n[CHECK] num_genes={num_genes}, "
              f"train_samples={num_samples}, val_samples={len(X_val)}")
        assert num_genes > 10000, f"Expected ~16K genes, got {num_genes}"

    # ─────────────────────────────────────────────────────────
    # DATASETS & DATALOADERS
    # ─────────────────────────────────────────────────────────
    train_ds = ExpressionMLMDataset(X_train, CONFIG['mask_ratio'], CONFIG['mask_token'])
    val_ds = ExpressionMLMDataset(X_val, CONFIG['mask_ratio'], CONFIG['mask_token'])

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                        rank=rank, shuffle=True, seed=42)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                      rank=rank, shuffle=False, seed=42)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              sampler=train_sampler, num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                            sampler=val_sampler, num_workers=0,
                            pin_memory=True)

    if is_main:
        print(f"\n[DATA] Train: {len(train_ds):,} samples, {len(train_loader)} batches")
        print(f"[DATA] Val:   {len(val_ds):,} samples, {len(val_loader)} batches")

    # ─────────────────────────────────────────────────────────
    # MODEL
    # ─────────────────────────────────────────────────────────
    if is_main:
        print("\n[MODEL] Building ExpressionPerformer...")

    model = ExpressionPerformer(
        num_genes=num_genes,
        hidden_dim=CONFIG['hidden_dim'],
        n_heads=CONFIG['num_heads'],
        n_layers=CONFIG['num_layers'],
        ffn_dim=CONFIG['ffn_dim'],
        ree_base=CONFIG['ree_base'],
        mask_token_id=CONFIG['mask_token'],
        feature_type=CONFIG['feature_type'],
        compute_type=CONFIG['compute_type'],
    ).to(device)

    model = DDP(model, device_ids=[rank], output_device=rank,
                find_unused_parameters=False)

    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"  ✓ Parameters: {total_params:,}")

    # ─────────────────────────────────────────────────────────
    # OPTIMIZER & SCHEDULER
    # ─────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                      weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'])

    if is_main:
        print(f"  ✓ AdamW (lr={CONFIG['learning_rate']})")

    # ─────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────
    if is_main:
        print("\n" + "=" * 70)
        print("[TRAIN] Starting training...")
        print("=" * 70 + "\n")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    ckpt_base = Path(CONFIG['checkpoint_dir'])
    # Per-run subdir (wandb run ID if available, else timestamp)
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    if is_main and HAS_WANDB and wandb.run is not None:
        run_id = wandb.run.id
    else:
        run_id = run_timestamp
    run_tag = build_run_tag(CONFIG)
    ckpt_dir = ckpt_base / run_id
    if is_main:
        ckpt_base.mkdir(exist_ok=True, parents=True)
        ckpt_dir.mkdir(exist_ok=True, parents=True)

    # Load global best val loss (across all runs)
    global_best_path = ckpt_base / 'global_best_val_loss.json'
    if global_best_path.exists():
        with open(global_best_path) as f:
            global_best_val_loss = json.load(f)['val_loss']
    else:
        global_best_val_loss = float('inf')

    run_metadata = {
        'run_id': run_id,
        'timestamp': run_timestamp,
        'run_tag': run_tag,
        'normalization': CONFIG['normalization'],
        'sweep_parameters': {
            'learning_rate': CONFIG['learning_rate'],
            'weight_decay': CONFIG['weight_decay'],
            'mask_ratio': CONFIG['mask_ratio'],
            'ree_base': CONFIG['ree_base'],
            'early_stopping': CONFIG['early_stopping'],
        },
        'architecture': {
            'hidden_dim': CONFIG['hidden_dim'],
            'ffn_dim': CONFIG['ffn_dim'],
            'num_heads': CONFIG['num_heads'],
            'num_layers': CONFIG['num_layers'],
        },
        'dataset': {
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'num_genes': int(num_genes),
            'train_used_counts': train_used_counts,
            'val_used_counts': val_used_counts,
            'train_raw_counts': train_raw_counts,
            'val_raw_counts': val_raw_counts,
            'balanced_sampling': CONFIG['balanced_sampling'],
            'train_subset': CONFIG['train_subset'],
            'val_subset': CONFIG['val_subset'],
        },
    }

    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)

        # --- Train ---
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (x_masked, x_true, mask_idx) in enumerate(train_loader):
            x_masked = x_masked.to(device)
            x_true = x_true.to(device)

            pred = model(x_masked)  # [B, G]

            # MSE loss on masked positions only
            loss_parts = []
            for i in range(len(x_masked)):
                idxs = mask_idx[i]
                if len(idxs) > 0:
                    loss_parts.append(F.mse_loss(pred[i, idxs], x_true[i, idxs]))

            loss = torch.stack(loss_parts).mean() if loss_parts else torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            # Progress every 25%
            if is_main and (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                avg = running_loss / num_batches
                print(f"  Epoch {epoch+1}/{CONFIG['epochs']} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.6f} | Avg: {avg:.6f}")

        epoch_train_loss = running_loss / max(1, num_batches)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x_masked, x_true, mask_idx in val_loader:
                x_masked = x_masked.to(device)
                x_true = x_true.to(device)
                pred = model(x_masked)

                loss_parts = []
                for i in range(len(x_masked)):
                    idxs = mask_idx[i]
                    if len(idxs) > 0:
                        loss_parts.append(F.mse_loss(pred[i, idxs], x_true[i, idxs]))

                if loss_parts:
                    val_loss += torch.stack(loss_parts).mean().item()
                    val_batches += 1

        # Sync validation across ranks
        vl = torch.tensor(val_loss, device=device)
        vb = torch.tensor(float(val_batches), device=device)
        dist.all_reduce(vl, op=dist.ReduceOp.SUM)
        dist.all_reduce(vb, op=dist.ReduceOp.SUM)
        epoch_val_loss = (vl / vb.clamp(min=1)).item()

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        scheduler.step()

        # Log to wandb
        if is_main and HAS_WANDB:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'lr': scheduler.get_last_lr()[0],
            })

        epoch_time = time.time() - epoch_start

        # --- Checkpoint ---
        if is_main:
            model_sd = model.module.state_dict()

            print(f"\n  ╔════════════════════════════════════════════╗")
            print(f"  ║ Epoch {epoch+1}/{CONFIG['epochs']}")
            print(f"  ║ Train Loss: {epoch_train_loss:.6f}")
            print(f"  ║ Val Loss:   {epoch_val_loss:.6f}")
            print(f"  ║ Time: {epoch_time:.1f}s")

            checkpoint_payload = {
                'model_state_dict': model_sd,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'config': dict(CONFIG),
                'run_metadata': run_metadata,
                'total_params': total_params,
            }

            torch.save(checkpoint_payload, ckpt_dir / f"epoch_{epoch:02d}.pt")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(checkpoint_payload, ckpt_dir / "best_model.pt")
                best_named = ckpt_dir / f"best_{run_tag}_run-{run_id}.pt"
                torch.save(checkpoint_payload, best_named)
                print(f"  ║ ✓ New best (run)! Saved best_model.pt")

                # Update global best across all runs
                if epoch_val_loss < global_best_val_loss:
                    global_best_val_loss = epoch_val_loss
                    torch.save(checkpoint_payload, ckpt_base / "best_model.pt")
                    with open(global_best_path, 'w') as f:
                        json.dump({'val_loss': global_best_val_loss,
                                   'run_id': run_id,
                                   'epoch': epoch + 1,
                                   'run_tag': run_tag,
                                   'normalization': CONFIG['normalization']}, f, indent=2)
                    print(f"  ║ ★ New global best! {epoch_val_loss:.6f}")
            else:
                if CONFIG['early_stopping']:
                    patience_counter += 1
                    print(f"  ║ ✗ No improvement ({patience_counter}/{CONFIG['patience']})")
                    if patience_counter >= CONFIG['patience']:
                        print(f"  ║ ⚠ Early stopping!")
                        print(f"  ╚════════════════════════════════════════════╝\n")
                        break
                else:
                    print("  ║ ✗ No improvement (early_stopping=False; continuing)")

            print(f"  ╚════════════════════════════════════════════╝\n")

    # ─────────────────────────────────────────────────────────
    # SAVE ARTIFACTS
    # ─────────────────────────────────────────────────────────
    if is_main:
        # Config
        cfg = {
            **CONFIG,
            'num_genes': num_genes,
            'total_params': total_params,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1,
            'run_id': run_id,
            'timestamp': run_timestamp,
            'run_tag': run_tag,
            'dataset': run_metadata['dataset'],
            'architecture': run_metadata['architecture'],
            'sweep_parameters': run_metadata['sweep_parameters'],
        }
        with open(ckpt_dir / "config.json", 'w') as f:
            json.dump(cfg, f, indent=2)

        with open(ckpt_dir / "run_metadata.json", 'w') as f:
            json.dump(run_metadata, f, indent=2)

        # Loss CSV
        pd.DataFrame({'epoch': range(len(train_losses)),
                       'train_loss': train_losses,
                       'val_loss': val_losses}).to_csv(
            ckpt_dir / "loss_history.csv", index=False)

        # Loss plot
        if HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, marker='o', label='Train Loss', linewidth=2)
            plt.plot(val_losses, marker='s', label='Val Loss', linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.title("ExpressionPerformer Training")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(ckpt_dir / "loss_plot.png", dpi=150)
            plt.close()

        total_time = time.time() - script_start
        print("=" * 70)
        print(f"Training complete! {total_time:.0f}s ({total_time/60:.1f}m)")
        print(f"  Run best val loss:    {best_val_loss:.6f}")
        print(f"  Global best val loss: {global_best_val_loss:.6f}")
        print(f"  Run checkpoints:      {ckpt_dir}/")
        print(f"  Global best model:    {ckpt_base / 'best_model.pt'}")
        print("=" * 70 + "\n")

    if is_main and HAS_WANDB:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
