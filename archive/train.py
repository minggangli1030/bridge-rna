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

# Force unbuffered output for DDP visibility (safe for all ranks)
try:
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
except Exception as e:
    print(f"[WARN] Could not set unbuffered output: {e}", file=sys.stderr)

import time
import json
from pathlib import Path
from collections import OrderedDict
import bisect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.distributed as dist
import pandas as pd
import pyarrow.parquet as pq

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
    'normalization': 'log1p_tpm',      # 'tpm' or 'log1p_tpm' applied before REE/model input
    'mask_ratio': 0.15,
    'mask_token': -10,
    'learning_rate': 1e-4,
    'weight_decay': 0,
    'batch_size': 4,
    'epochs': 1,
    'early_stopping': True,
    'patience': 5,
    'seed': 42,
    # Data loading mode: 'preload' (load arrays into RAM) or 'streaming' (on-the-fly parquet reads)
    'data_mode': 'streaming',
    'stream_cache_size': 2,
    'num_workers': 1,
    'prefetch_factor': 2,
    'persistent_workers': False,
    # Data subset sizes (set to None for all available)
    'train_subset': 2000,
    'val_subset': 400,
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


class StreamingParquetMLMDataset(Dataset):
    """
    High-throughput parquet streaming dataset using PyArrow row-group reads.

    Performance features:
      - Avoids pandas/DataFrame conversion in the hot path.
      - Reads row groups instead of full files for sample access.
      - LRU cache stores decoded row-group arrays to reduce repeated I/O.
      - DDP-aware file sharding (files[rank::world_size]) to reduce contention.
    """

    def __init__(self, batch_dir, sample_indices, normalization='tpm', mask_ratio=0.15,
                 mask_token=-10, cache_size=16, rank=0, world_size=1,
                 ddp_file_split=True):
        self.batch_dir = Path(batch_dir)
        self.batch_files = sorted(self.batch_dir.glob("*.parquet"))
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.normalization = normalization
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token
        self.cache_size = max(1, int(cache_size))
        self._cache = OrderedDict()  # (batch_idx, row_group_idx) -> pyarrow.Table
        self._process_pid = None
        self._parquet_files = None

        # Build row-group cumulative row starts for fast row lookup.
        self._row_group_starts = []
        first_pf = None
        for file_idx, batch_file in enumerate(self.batch_files):
            pf = pq.ParquetFile(str(batch_file))
            if file_idx == 0:
                first_pf = pf
            starts = [0]
            for rg in range(pf.metadata.num_row_groups):
                starts.append(starts[-1] + pf.metadata.row_group(rg).num_rows)
            self._row_group_starts.append(starts)

        sample_indices = list(sample_indices)

        # DDP-aware file ownership: each rank gets a disjoint subset of files.
        if ddp_file_split and self.world_size > 1:
            my_files = list(range(len(self.batch_files)))[self.rank::self.world_size]
            my_file_set = set(my_files)
            self.sample_indices = [s for s in sample_indices if s[0] in my_file_set]
        else:
            self.sample_indices = sample_indices

        # Precompute per-sample row-group metadata once to avoid repeated bisect work.
        # record = (batch_idx, row_group_idx, row_offset)
        self.records = []
        self.group_to_indices = {}
        for i, (batch_idx, sample_row) in enumerate(self.sample_indices):
            rg_idx, rg_offset = self._locate_row_group(batch_idx, sample_row)
            self.records.append((batch_idx, rg_idx, rg_offset))
            key = (batch_idx, rg_idx)
            self.group_to_indices.setdefault(key, []).append(i)

        # Keep only gene columns in the streaming hot path.
        first_schema_cols = first_pf.schema_arrow.names
        self._gene_columns = [c for c in first_schema_cols if c not in ('geo_accession', '__index_level_0__')]
        self.num_genes = len(self._gene_columns)
        self.num_mask = max(1, int(self.num_genes * self.mask_ratio))

    def __len__(self):
        return len(self.sample_indices)

    def _table_to_numpy(self, table):
        # Convert Arrow table [rows, genes] to float32 NumPy for the selected rows only.
        cols = [table.column(i).combine_chunks().to_numpy(zero_copy_only=False)
                for i in range(table.num_columns)]
        return np.column_stack(cols).astype(np.float32, copy=False)

    def _locate_row_group(self, batch_idx, sample_row):
        starts = self._row_group_starts[batch_idx]
        rg_idx = bisect.bisect_right(starts, sample_row) - 1
        rg_offset = sample_row - starts[rg_idx]
        return rg_idx, rg_offset

    def _ensure_process_state(self):
        current_pid = os.getpid()
        if self._process_pid == current_pid and self._parquet_files is not None:
            return

        self._process_pid = current_pid
        self._cache = OrderedDict()
        self._parquet_files = [pq.ParquetFile(str(p)) for p in self.batch_files]

    def _get_row_group_table(self, batch_idx, row_group_idx):
        self._ensure_process_state()
        key = (batch_idx, row_group_idx)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        table = self._parquet_files[batch_idx].read_row_group(
            row_group_idx, columns=self._gene_columns, use_threads=True
        )
        self._cache[key] = table
        self._cache.move_to_end(key)

        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return table

    def __getitem__(self, idx):
        # Return lightweight record index; real data is loaded in collate_batch.
        return int(idx)

    def collate_batch(self, batch_record_indices):
        """
        Batch-level row-group loading.

        This reduces Python overhead and avoids repeatedly materializing row-group
        arrays per individual sample.
        """
        B = len(batch_record_indices)
        x_true = np.empty((B, self.num_genes), dtype=np.float32)

        # Group requests by (file, row_group) to maximize locality.
        grouped = {}
        for out_i, rec_i in enumerate(batch_record_indices):
            batch_idx, rg_idx, rg_off = self.records[rec_i]
            grouped.setdefault((batch_idx, rg_idx), []).append((out_i, rg_off))

        for (batch_idx, rg_idx), reqs in grouped.items():
            table = self._get_row_group_table(batch_idx, rg_idx)
            local_rows = [r for _, r in reqs]

            # Read only requested rows from this row-group.
            sub = table.take(np.array(local_rows, dtype=np.int64))
            sub_np = self._table_to_numpy(sub)
            for j, (out_i, _) in enumerate(reqs):
                x_true[out_i] = sub_np[j]

        if self.normalization == 'log1p_tpm':
            x_true = np.log1p(np.maximum(x_true, 0.0)).astype(np.float32, copy=False)

        mask_indices = np.empty((B, self.num_mask), dtype=np.int64)
        for i in range(B):
            mask_indices[i] = np.random.choice(self.num_genes, self.num_mask, replace=False)

        x_masked = x_true.copy()
        x_masked[np.arange(B)[:, None], mask_indices] = self.mask_token

        return (
            torch.from_numpy(x_masked),
            torch.from_numpy(x_true),
            torch.from_numpy(mask_indices),
        )


class RowGroupBatchSampler(Sampler):
    """Batch sampler that keeps batches within the same (file, row_group)."""

    def __init__(self, group_to_indices, batch_size, shuffle=True, seed=42, drop_last=False):
        self.group_to_indices = group_to_indices
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        total = 0
        for idxs in self.group_to_indices.values():
            if self.drop_last:
                total += len(idxs) // self.batch_size
            else:
                total += (len(idxs) + self.batch_size - 1) // self.batch_size
        self._len = total

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        keys = list(self.group_to_indices.keys())
        if self.shuffle:
            rng.shuffle(keys)

        for k in keys:
            idxs = np.array(self.group_to_indices[k], dtype=np.int64)
            if self.shuffle:
                rng.shuffle(idxs)

            n = len(idxs)
            stop = (n // self.batch_size) * self.batch_size if self.drop_last else n
            for start in range(0, stop, self.batch_size):
                batch = idxs[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch.tolist()

    def __len__(self):
        return self._len


# ============================================================
# DATA LOADING
# ============================================================
def get_sample_indices(batch_dir, train_subset=None, val_subset=None, balanced_sampling=True, seed=42, verbose=True):
    """
    Build sample index lists for train/val without loading all data.
    
    Args:
        batch_dir: Path to batch parquet files
        train_subset: Exact number of train samples to select (None = use all available)
        val_subset: Exact number of val samples to select (None = use remaining after train)
        balanced_sampling: If True, balance human/mouse to min count
        seed: Random seed
        verbose: Print diagnostics
    
    Returns:
        (train_sample_indices, val_sample_indices): Lists of
        (batch_idx, sample_idx_in_batch) tuples where sample_idx_in_batch is row index.
    """
    batch_dir = Path(batch_dir)
    batch_files = sorted(batch_dir.glob("*.parquet"))
    
    if not batch_files:
        # Final debug before error
        print(f"[ERROR] No parquet files found in {batch_dir}")
        print(f"[DEBUG] Directory exists: {batch_dir.exists()}")
        print(f"[DEBUG] Is directory: {batch_dir.is_dir()}")
        if batch_dir.exists():
            print(f"[DEBUG] Files in directory: {list(batch_dir.iterdir())[:10]}")
        raise FileNotFoundError(f"No parquet files found in {batch_dir}")
    
    if verbose:
        print(f"[DEBUG] Found {len(batch_files)} batch files")
    
    # Load metadata to track species per sample
    metadata_file = batch_dir.parent / "samples.json"
    sample_to_species = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            samples_meta = json.load(f)
        sample_to_species = {s["id"]: s["species"] for s in samples_meta if "species" in s}
    
    rng = np.random.default_rng(seed)
    
    # Build master list of all (batch_idx, sample_in_batch, species) tuples.
    # New preprocessing saves sample-major batch files, so sample IDs are parquet index.
    all_samples = []  # [(batch_idx, sample_in_batch, species), ...]

    manifest_path = batch_dir.parent / "batch_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            batch_manifest = json.load(f)

        # Prefer direct filename lookup; fallback to positional lists for legacy
        # manifests that use synthetic keys like batch_0001.parquet.
        ordered_manifest_lists = [batch_manifest[k] for k in sorted(batch_manifest.keys())]

        for batch_idx, batch_file in enumerate(batch_files):
            sample_ids = batch_manifest.get(batch_file.name)
            if sample_ids is None and batch_idx < len(ordered_manifest_lists):
                sample_ids = ordered_manifest_lists[batch_idx]
            if sample_ids is None:
                sample_ids = []
            for sample_idx, sample_id in enumerate(sample_ids):
                species = sample_to_species.get(sample_id, "unknown")
                all_samples.append((batch_idx, sample_idx, species))

        # If manifest exists but produced no sample rows, fallback to parquet index.
        if not all_samples:
            for batch_idx, batch_file in enumerate(batch_files):
                pf = pq.ParquetFile(str(batch_file))
                cols = pf.schema_arrow.names
                idx_col = 'geo_accession' if 'geo_accession' in cols else (
                    '__index_level_0__' if '__index_level_0__' in cols else None
                )
                if idx_col is not None:
                    table = pf.read(columns=[idx_col], use_threads=True)
                    sample_ids = table.column(0).to_pylist()
                else:
                    sample_ids = [str(i) for i in range(pf.metadata.num_rows)]
                for sample_idx, sample_id in enumerate(sample_ids):
                    species = sample_to_species.get(sample_id, "unknown")
                    all_samples.append((batch_idx, sample_idx, species))
    else:
        # Fallback for legacy data without manifest: read index column via PyArrow.
        for batch_idx, batch_file in enumerate(batch_files):
            pf = pq.ParquetFile(str(batch_file))
            cols = pf.schema_arrow.names
            idx_col = 'geo_accession' if 'geo_accession' in cols else (
                '__index_level_0__' if '__index_level_0__' in cols else None
            )
            if idx_col is not None:
                table = pf.read(columns=[idx_col], use_threads=True)
                sample_ids = table.column(0).to_pylist()
            else:
                sample_ids = [str(i) for i in range(pf.metadata.num_rows)]
            for sample_idx, sample_id in enumerate(sample_ids):
                species = sample_to_species.get(sample_id, "unknown")
                all_samples.append((batch_idx, sample_idx, species))
    
    if verbose:
        print(f"[DATA] Total samples available: {len(all_samples):,}", flush=True)
    
    # Separate by species
    samples_by_species = {}
    for batch_idx, sample_idx, species in all_samples:
        if species not in samples_by_species:
            samples_by_species[species] = []
        samples_by_species[species].append((batch_idx, sample_idx))
    
    if verbose:
        for sp, samples in samples_by_species.items():
            print(f"       {sp}: {len(samples):,} samples", flush=True)
    
    # Apply balanced sampling and subsetting
    if balanced_sampling and len(samples_by_species) > 1:
        # Determine total requested before split.
        requested_total = None
        if train_subset is not None and val_subset is not None:
            requested_total = train_subset + val_subset
        elif train_subset is not None:
            requested_total = train_subset

        # Determine per-species limit.
        if requested_total is not None:
            per_species = requested_total // len(samples_by_species)
        else:
            # Use max fully balanced pool based on minority species.
            per_species = min(len(samples) for samples in samples_by_species.values())
        
        if verbose:
            print(f"       Balanced to {per_species:,} per species", flush=True)
        
        all_samples_balanced = []
        for species, samples in samples_by_species.items():
            if len(samples) > per_species:
                selected = rng.choice(len(samples), per_species, replace=False)
                all_samples_balanced.extend([samples[i] for i in selected])
            else:
                all_samples_balanced.extend(samples)
        all_samples = all_samples_balanced
    
    elif train_subset is not None:
        # No species balancing, just subsample
        if val_subset is not None:
            requested_total = train_subset + val_subset
        else:
            requested_total = train_subset
        requested_total = min(requested_total, len(all_samples))
        selected = rng.choice(len(all_samples), requested_total, replace=False)
        all_samples = [all_samples[i] for i in selected]

    # Final shuffled pool from which we take exact train/val counts.
    indices = np.arange(len(all_samples))
    rng.shuffle(indices)
    shuffled = [all_samples[i] for i in indices]

    if train_subset is None:
        # Backward-compatible default when no explicit train size is provided.
        train_count = int(0.8 * len(shuffled))
    else:
        train_count = min(train_subset, len(shuffled))

    remaining = max(0, len(shuffled) - train_count)
    if val_subset is None:
        val_count = remaining
    else:
        val_count = min(val_subset, remaining)

    train_indices = shuffled[:train_count]
    val_indices = shuffled[train_count:train_count + val_count]
    
    if verbose:
        print(f"       Train: {len(train_indices):,} samples", flush=True)
        print(f"       Val:   {len(val_indices):,} samples", flush=True)
    
    return train_indices, val_indices


def load_batch_data(batch_dir, sample_indices, normalization='tpm', verbose=True):
    """
    Load selected samples from batch parquet files into a single numpy array.
    
    Args:
        batch_dir: Path to directory with batch parquet files
        sample_indices: List of (batch_idx, sample_idx_in_batch) tuples
        normalization: 'tpm' or 'log1p_tpm'
        verbose: Print progress
    
    Returns:
        numpy array of shape [num_samples, num_genes]
    """
    batch_dir = Path(batch_dir)
    batch_files = sorted(batch_dir.glob("*.parquet"))
    
    # Group samples by batch file for efficient loading
    from collections import defaultdict
    batch_to_samples = defaultdict(list)
    for idx, (batch_idx, sample_in_batch) in enumerate(sample_indices):
        batch_to_samples[batch_idx].append((idx, sample_in_batch))
    
    # Pre-allocate output array (sample-major parquet: [samples, genes]).
    first_pf = pq.ParquetFile(str(batch_files[0]))
    gene_cols = [c for c in first_pf.schema_arrow.names
                 if c not in ('geo_accession', '__index_level_0__')]
    num_genes = len(gene_cols)
    result = np.empty((len(sample_indices), num_genes), dtype=np.float32)
    
    # Load batch-by-batch and gather selected sample rows.
    total_batches = len(batch_to_samples)
    for i, (batch_idx, idx_pairs) in enumerate(batch_to_samples.items(), start=1):
        table = pq.read_table(batch_files[batch_idx], columns=gene_cols, use_threads=True)
        cols = [table.column(j).combine_chunks().to_numpy(zero_copy_only=False)
                for j in range(table.num_columns)]
        data = np.stack(cols, axis=1).astype(np.float32, copy=False)
        for out_idx, sample_in_batch in idx_pairs:
            result[out_idx] = data[sample_in_batch]

        if verbose and (i % 25 == 0 or i == total_batches):
            print(f"  ...loaded {i}/{total_batches} batch files", flush=True)
    
    # Apply normalization
    if normalization == 'log1p_tpm':
        result = np.log1p(np.maximum(result, 0.0)).astype(np.float32)
    
    if verbose:
        print(f"  ✓ Loaded {result.shape[0]:,} samples × {result.shape[1]:,} genes")
    
    return result


def get_num_genes_from_batches(batch_dir):
    """Infer number of genes from sample-major batch parquet shape."""
    batch_files = sorted(Path(batch_dir).glob("*.parquet"))
    if not batch_files:
        raise FileNotFoundError(f"No parquet files found in {batch_dir}")
    pf = pq.ParquetFile(str(batch_files[0]))
    cols = [c for c in pf.schema_arrow.names if c not in ('geo_accession', '__index_level_0__')]
    return len(cols)


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


# ============================================================
# TRAINING (DDP)
# ============================================================
def main():
    print("\n[STARTUP] train.py started - initializing DDP...", flush=True)
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
        print(f"ExpressionPerformer Training — DDP ({world_size} processes)")
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
        # Always derive ffn_dim from hidden_dim (4x multiplier)
        CONFIG['ffn_dim'] = CONFIG['hidden_dim'] * 4
        wandb.config.update(CONFIG, allow_val_change=True)

    # Broadcast CONFIG from rank 0 so all ranks use the same hyperparams
    config_list = [CONFIG if is_main else None]
    dist.broadcast_object_list(config_list, src=0)
    CONFIG.update(config_list[0])

    # ─────────────────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────────────────
    data_dir = Path(CONFIG['data_dir'])
    batch_dir = data_dir / "batch_files"
    if not batch_dir.exists():
        batch_dir = data_dir

    if is_main:
        print("\n[DATA] Building sample indices...", flush=True)

    t0 = time.time()
    train_indices = None
    val_indices = None
    if is_main:
        train_indices, val_indices = get_sample_indices(
            batch_dir,
            train_subset=CONFIG.get('train_subset', None),
            val_subset=CONFIG.get('val_subset', None),
            balanced_sampling=CONFIG.get('balanced_sampling', True),
            seed=CONFIG['seed'],
            verbose=True,
        )

    train_indices_list = [train_indices if is_main else None]
    val_indices_list = [val_indices if is_main else None]
    dist.broadcast_object_list(train_indices_list, src=0)
    dist.broadcast_object_list(val_indices_list, src=0)
    train_indices = train_indices_list[0]
    val_indices = val_indices_list[0]
    if is_main:
        print(f"  ✓ Index time: {time.time()-t0:.1f}s", flush=True)

    data_mode = CONFIG.get('data_mode', 'preload')
    if data_mode == 'streaming':
        if is_main:
            print("\n[DATA] Using streaming mode (on-the-fly parquet reads)", flush=True)

        train_ds = StreamingParquetMLMDataset(
            batch_dir,
            train_indices,
            normalization=CONFIG['normalization'],
            mask_ratio=CONFIG['mask_ratio'],
            mask_token=CONFIG['mask_token'],
            cache_size=CONFIG.get('stream_cache_size', 2),
            rank=rank,
            world_size=world_size,
            ddp_file_split=True,
        )
        val_ds = StreamingParquetMLMDataset(
            batch_dir,
            val_indices,
            normalization=CONFIG['normalization'],
            mask_ratio=CONFIG['mask_ratio'],
            mask_token=CONFIG['mask_token'],
            cache_size=CONFIG.get('stream_cache_size', 2),
            rank=rank,
            world_size=world_size,
            ddp_file_split=True,
        )
        num_genes = get_num_genes_from_batches(batch_dir)
    else:
        if is_main:
            print("\n[DATA] Loading training data into memory...", flush=True)
        X_train = load_batch_data(batch_dir, train_indices,
                                  normalization=CONFIG['normalization'],
                                  verbose=is_main)
        if is_main:
            print("[DATA] Loading validation data into memory...", flush=True)
        X_val = load_batch_data(batch_dir, val_indices,
                                normalization=CONFIG['normalization'],
                                verbose=is_main)

        num_genes = X_train.shape[1]

        # Data stored fully in host memory; faster per-step but higher RAM.
        train_ds = ExpressionMLMDataset(X_train, CONFIG['mask_ratio'], CONFIG['mask_token'])
        val_ds = ExpressionMLMDataset(X_val, CONFIG['mask_ratio'], CONFIG['mask_token'])

    if is_main:
        print(f"\n[CHECK] num_genes={num_genes}")
        assert num_genes > 10000, f"Expected ~16K genes, got {num_genes}"

    # ─────────────────────────────────────────────────────────
    # DATASETS & DATALOADERS
    # ─────────────────────────────────────────────────────────
    if data_mode == 'streaming':
        train_batch_sampler = RowGroupBatchSampler(
            train_ds.group_to_indices,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            seed=CONFIG.get('seed', 42),
            drop_last=False,
        )
        val_batch_sampler = RowGroupBatchSampler(
            val_ds.group_to_indices,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            seed=CONFIG.get('seed', 42),
            drop_last=False,
        )
    else:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                            rank=rank, shuffle=True, seed=42)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                          rank=rank, shuffle=False, seed=42)

    num_workers = int(CONFIG.get('num_workers', 0))
    if data_mode == 'streaming' and num_workers > 0:
        if is_main:
            print("[DATA] Streaming mode forcing num_workers=0 to avoid per-worker row-group cache OOM.", flush=True)
        num_workers = 0

    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = int(CONFIG.get('prefetch_factor', 2))
        loader_kwargs['persistent_workers'] = bool(CONFIG.get('persistent_workers', False))

    if data_mode == 'streaming':
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            collate_fn=train_ds.collate_batch,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_batch_sampler,
            collate_fn=val_ds.collate_batch,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                                  sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                                sampler=val_sampler, **loader_kwargs)

    if is_main:
        print(f"\n[DATA] Train: {len(train_ds):,} samples, {len(train_loader)} batches")
        print(f"[DATA] Val:   {len(val_ds):,} samples, {len(val_loader)} batches")

    # Synchronize after data loading
    dist.barrier()

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
        print("\n" + "=" * 70, flush=True)
        print("[TRAIN] Starting training...", flush=True)
        print("=" * 70 + "\n", flush=True)

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
            'train_samples': len(train_ds),
            'val_samples': len(val_ds),
            'num_genes': int(num_genes),
            'train_used_counts': None,  # Not computed for lazy-loaded data
            'val_used_counts': None,
            'train_raw_counts': None,
            'val_raw_counts': None,
            'balanced_sampling': CONFIG['balanced_sampling'],
            'train_subset': CONFIG['train_subset'],
            'val_subset': CONFIG['val_subset'],
        },
    }

    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        if data_mode == 'streaming':
            train_batch_sampler.set_epoch(epoch)
        else:
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

    # Ensure all ranks finish before cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Exception in train.py: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Try to cleanup DDP even on error
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass
        sys.exit(1)
