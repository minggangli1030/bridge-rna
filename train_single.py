#!/usr/bin/env python3
# coding=utf-8

"""
ExpressionBERT training for a single expression.parquet file.

This keeps the same model/training behavior as train.py while removing
batch-file orchestration. Data loading is always streaming and row-group aware.

Usage:
  torchrun --nproc_per_node=2 train_single.py
"""

import os
import sys

# Force unbuffered output for DDP visibility (safe for all ranks)
try:
	sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
	sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
except Exception as e:
	print(f"[WARN] Could not set unbuffered output: {e}", file=sys.stderr)

import bisect
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler, Sampler

from slim_performer_model import SLiMPerformerLayer

try:
	import wandb

	HAS_WANDB = True
except ImportError:
	HAS_WANDB = False

try:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	HAS_MATPLOTLIB = True
except ImportError:
	HAS_MATPLOTLIB = False


CONFIG = {
	"hidden_dim": 768,
	"ffn_dim": 3072,
	"num_heads": 8,
	"num_layers": 2,
	"ree_base": 100.0,
	"feature_type": "sqr",
	"compute_type": "iter",
	"normalization": "log1p_tpm",
	"mask_ratio": 0.15,
	"mask_token": -10,
	"learning_rate": 2e-4,
	"weight_decay": 0,
	"batch_size": 4,
	"epochs": 30,
	"early_stopping": True,
	"patience": 5,
	"seed": 42,
	# Streaming defaults chosen for stability on large single parquet files.
	"data_mode": "streaming",
	"stream_cache_size": 2,
	"num_workers": 1,
	"prefetch_factor": 2,
	"persistent_workers": False,
	# Exact split sizes (None = use all remaining).
	"train_subset": 20000,
	"val_subset": 4000,
	"balanced_sampling": True,
	"expression_parquet": "./data/archs4/train_orthologs_merged/expression.parquet",
	"samples_json": "./data/archs4/train_orthologs/samples.json",
	"checkpoint_dir": "./checkpoints_performer_single",
	# Memory optimizations
	"use_amp": True,                   # mixed precision (fp16) — ~2x memory reduction
	"gradient_checkpointing": True,    # recompute activations during backward — trades compute for memory
}


class RotaryExpressionEmbedding(nn.Module):
	def __init__(self, dim, base=100.0, mask_token_id=-10):
		super().__init__()
		self.dim = dim
		self.mask_token_id = mask_token_id
		inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
		self.register_buffer("inv_freq", inv_freq)

	def forward(self, x):
		x_mask_idx = (x == self.mask_token_id).nonzero(as_tuple=False)
		freqs = torch.einsum("bi,j->bij", x, self.inv_freq)
		emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
		if len(x_mask_idx) > 0:
			emb[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = 0
		return emb


class ExpressionPerformer(nn.Module):
	def __init__(
		self,
		num_genes,
		hidden_dim=256,
		n_heads=8,
		n_layers=4,
		ffn_dim=1024,
		ree_base=100.0,
		mask_token_id=-10,
		feature_type="sqr",
		compute_type="iter",
		gradient_checkpointing=False,
	):
		super().__init__()
		self.num_genes = num_genes
		self.gradient_checkpointing = gradient_checkpointing
		self.gene_embedding = nn.Embedding(num_genes, hidden_dim)
		self.ree = RotaryExpressionEmbedding(hidden_dim, base=ree_base, mask_token_id=mask_token_id)
		self.layers = nn.ModuleList(
			[
				SLiMPerformerLayer(hidden_dim, ffn_dim, n_heads, feature_type, compute_type, on_gptln=True)
				for _ in range(n_layers)
			]
		)
		self.output_map = nn.Linear(hidden_dim, 1)

	def forward(self, x):
		_, g = x.shape
		device = x.device
		gene_ids = torch.arange(g, device=device)
		gene_emb = self.gene_embedding(gene_ids)
		ree_emb = self.ree(x)
		h = gene_emb.unsqueeze(0) + ree_emb
		for layer in self.layers:
			rfs = layer.attention.sample_rfs(device)
			if self.gradient_checkpointing and self.training:
				h = torch.utils.checkpoint.checkpoint(layer.full_forward, h, rfs, use_reentrant=False)
			else:
				h = layer.full_forward(h, rfs)
		return self.output_map(h).squeeze(-1)


class SingleParquetStreamingMLMDataset(Dataset):
	"""Streaming dataset for a single sample-major parquet file."""

	def __init__(
		self,
		parquet_path,
		selected_rows,
		normalization="tpm",
		mask_ratio=0.15,
		mask_token=-10,
		cache_size=2,
	):
		self.parquet_path = Path(parquet_path)
		self.selected_rows = list(selected_rows)
		self.normalization = normalization
		self.mask_ratio = mask_ratio
		self.mask_token = mask_token
		self.cache_size = max(1, int(cache_size))

		pf = pq.ParquetFile(str(self.parquet_path))
		cols = pf.schema_arrow.names
		self.idx_col = next((c for c in ("geo_accession", "__index_level_0__", "sample_id") if c in cols), None)
		# idx_col is None means all columns are gene columns (no explicit sample ID)

		self.gene_cols = [c for c in cols if c not in ("geo_accession", "__index_level_0__", "sample_id")]
		self.num_genes = len(self.gene_cols)
		self.num_mask = max(1, int(self.num_genes * self.mask_ratio))

		starts = [0]
		for rg in range(pf.metadata.num_row_groups):
			starts.append(starts[-1] + pf.metadata.row_group(rg).num_rows)
		self.row_group_starts = starts

		self.records = []
		self.group_to_indices = {}
		for i, global_row in enumerate(self.selected_rows):
			rg_idx = bisect.bisect_right(self.row_group_starts, global_row) - 1
			rg_offset = global_row - self.row_group_starts[rg_idx]
			self.records.append((rg_idx, rg_offset))
			self.group_to_indices.setdefault(rg_idx, []).append(i)

		# Process-local parquet handles/cache to avoid worker-fork surprises.
		self._process_pid = None
		self._pf = None
		self._cache = OrderedDict()

	def __len__(self):
		return len(self.records)

	def __getitem__(self, idx):
		return int(idx)

	def _ensure_process_state(self):
		current_pid = os.getpid()
		if self._process_pid == current_pid and self._pf is not None:
			return
		self._process_pid = current_pid
		self._pf = pq.ParquetFile(str(self.parquet_path))
		self._cache = OrderedDict()

	def _get_row_group_table(self, rg_idx):
		self._ensure_process_state()
		if rg_idx in self._cache:
			self._cache.move_to_end(rg_idx)
			return self._cache[rg_idx]

		table = self._pf.read_row_group(rg_idx, columns=self.gene_cols, use_threads=True)
		self._cache[rg_idx] = table
		self._cache.move_to_end(rg_idx)
		if len(self._cache) > self.cache_size:
			self._cache.popitem(last=False)
		return table

	@staticmethod
	def _table_to_numpy(table):
		cols = [table.column(i).combine_chunks().to_numpy(zero_copy_only=False) for i in range(table.num_columns)]
		return np.column_stack(cols).astype(np.float32, copy=False)

	def collate_batch(self, batch_record_indices):
		b = len(batch_record_indices)
		x_true = np.empty((b, self.num_genes), dtype=np.float32)

		grouped = {}
		for out_i, rec_i in enumerate(batch_record_indices):
			rg_idx, rg_off = self.records[rec_i]
			grouped.setdefault(rg_idx, []).append((out_i, rg_off))

		for rg_idx, reqs in grouped.items():
			table = self._get_row_group_table(rg_idx)
			local_rows = [r for _, r in reqs]
			sub = table.take(pa.array(local_rows, type=pa.int64()))
			sub_np = self._table_to_numpy(sub)
			for j, (out_i, _) in enumerate(reqs):
				x_true[out_i] = sub_np[j]

		if self.normalization == "log1p_tpm":
			x_true = np.log1p(np.maximum(x_true, 0.0)).astype(np.float32, copy=False)

		mask_idx = np.empty((b, self.num_mask), dtype=np.int64)
		for i in range(b):
			mask_idx[i] = np.random.choice(self.num_genes, self.num_mask, replace=False)

		x_masked = x_true.copy()
		x_masked[np.arange(b)[:, None], mask_idx] = self.mask_token

		return torch.from_numpy(x_masked), torch.from_numpy(x_true), torch.from_numpy(mask_idx)


class InMemoryExpressionMLMDataset(Dataset):
	"""In-memory dataset for a selected subset of rows from one parquet file."""

	def __init__(self, x_data, normalization="tpm", mask_ratio=0.15, mask_token=-10):
		x_data = np.asarray(x_data, dtype=np.float32)
		if normalization == "log1p_tpm":
			x_data = np.log1p(np.maximum(x_data, 0.0)).astype(np.float32, copy=False)

		self.x_data = x_data
		self.normalization = normalization
		self.mask_ratio = mask_ratio
		self.mask_token = mask_token
		self.num_genes = int(self.x_data.shape[1])
		self.num_mask = max(1, int(self.num_genes * self.mask_ratio))

	def __len__(self):
		return int(self.x_data.shape[0])

	def __getitem__(self, idx):
		return int(idx)

	def collate_batch(self, batch_record_indices):
		rows = np.asarray(batch_record_indices, dtype=np.int64)
		x_true = self.x_data[rows]
		mask_idx = np.empty((len(rows), self.num_mask), dtype=np.int64)
		for i in range(len(rows)):
			mask_idx[i] = np.random.choice(self.num_genes, self.num_mask, replace=False)

		x_masked = x_true.copy()
		x_masked[np.arange(len(rows))[:, None], mask_idx] = self.mask_token
		return torch.from_numpy(x_masked), torch.from_numpy(x_true), torch.from_numpy(mask_idx)


class DistributedRowGroupBatchSampler(Sampler):
	"""DDP-safe row-group sampler: locality-aware and equal batch counts on all ranks."""

	def __init__(self, group_to_indices, batch_size, num_replicas, rank, shuffle=True, seed=42, drop_last=False):
		self.group_to_indices = group_to_indices
		self.batch_size = int(batch_size)
		self.num_replicas = int(num_replicas)
		self.rank = int(rank)
		self.shuffle = bool(shuffle)
		self.seed = int(seed)
		self.drop_last = bool(drop_last)
		self.epoch = 0

	def set_epoch(self, epoch):
		self.epoch = int(epoch)

	def _build_global_batches(self):
		rng = np.random.default_rng(self.seed + self.epoch)
		group_ids = list(self.group_to_indices.keys())
		if self.shuffle:
			rng.shuffle(group_ids)

		batches = []
		for g in group_ids:
			idxs = list(self.group_to_indices[g])
			if self.shuffle:
				rng.shuffle(idxs)
			for i in range(0, len(idxs), self.batch_size):
				batch = idxs[i : i + self.batch_size]
				if len(batch) < self.batch_size and self.drop_last:
					continue
				batches.append(batch)

		if not batches:
			return []

		if self.drop_last:
			total = (len(batches) // self.num_replicas) * self.num_replicas
			batches = batches[:total]
		else:
			rem = len(batches) % self.num_replicas
			if rem != 0:
				pad = self.num_replicas - rem
				batches.extend(batches[:pad])

		return batches

	def __iter__(self):
		global_batches = self._build_global_batches()
		for batch in global_batches[self.rank :: self.num_replicas]:
			yield batch

	def __len__(self):
		global_batches = self._build_global_batches()
		return len(global_batches[self.rank :: self.num_replicas])


def _load_sample_species(samples_json_path):
	p = Path(samples_json_path)
	if not p.exists():
		return {}
	with open(p) as f:
		rows = json.load(f)
	return {str(r["id"]): r.get("species", "unknown") for r in rows if "id" in r}


def _read_parquet_index_ids(parquet_path):
	pf = pq.ParquetFile(str(parquet_path))
	cols = pf.schema_arrow.names
	idx_col = next((c for c in ("geo_accession", "__index_level_0__", "sample_id") if c in cols), None)
	if idx_col is None:
		# No explicit index column — use integer row indices as sample IDs
		num_rows = pf.metadata.num_rows
		return [str(i) for i in range(num_rows)]
	t = pf.read(columns=[idx_col], use_threads=True)
	return [str(x) for x in t.column(0).to_pylist()]


def build_single_parquet_split(
	parquet_path,
	samples_json_path,
	train_subset=None,
	val_subset=None,
	balanced_sampling=True,
	seed=42,
	verbose=True,
):
	rng = np.random.default_rng(seed)
	sample_ids = _read_parquet_index_ids(parquet_path)
	species_map = _load_sample_species(samples_json_path)

	# (global_row_idx, species)
	all_rows = [(i, species_map.get(sid, "unknown")) for i, sid in enumerate(sample_ids)]

	if verbose:
		print(f"[DATA] Total samples available: {len(all_rows):,}", flush=True)

	by_species = {}
	for row_idx, species in all_rows:
		by_species.setdefault(species, []).append(row_idx)

	if verbose:
		for sp, rows in by_species.items():
			print(f"       {sp}: {len(rows):,} samples", flush=True)

	# Balance only over known species and only when more than one known species exists.
	known_species = {k: v for k, v in by_species.items() if k != "unknown"}
	if balanced_sampling and len(known_species) > 1:
		requested_total = None
		if train_subset is not None and val_subset is not None:
			requested_total = train_subset + val_subset
		elif train_subset is not None:
			requested_total = train_subset

		if requested_total is not None:
			per_species = requested_total // len(known_species)
		else:
			per_species = min(len(v) for v in known_species.values())

		selected_rows = []
		for sp, rows in known_species.items():
			if len(rows) > per_species:
				take = rng.choice(len(rows), per_species, replace=False)
				selected_rows.extend([rows[i] for i in take])
			else:
				selected_rows.extend(rows)

		all_pool = selected_rows
		if verbose:
			print(f"       Balanced to {per_species:,} per species", flush=True)
	else:
		all_pool = [r for r, _ in all_rows]

	# Optional non-balanced global subsampling.
	if not (balanced_sampling and len(known_species) > 1) and train_subset is not None:
		req_total = train_subset + val_subset if val_subset is not None else train_subset
		req_total = min(req_total, len(all_pool))
		take = rng.choice(len(all_pool), req_total, replace=False)
		all_pool = [all_pool[i] for i in take]

	all_pool = list(all_pool)
	rng.shuffle(all_pool)

	train_count = int(0.8 * len(all_pool)) if train_subset is None else min(train_subset, len(all_pool))
	remaining = max(0, len(all_pool) - train_count)
	val_count = remaining if val_subset is None else min(val_subset, remaining)

	train_rows = all_pool[:train_count]
	val_rows = all_pool[train_count : train_count + val_count]

	if verbose:
		print(f"       Train: {len(train_rows):,} samples", flush=True)
		print(f"       Val:   {len(val_rows):,} samples", flush=True)

	return train_rows, val_rows


def get_num_genes_from_single_parquet(parquet_path):
	pf = pq.ParquetFile(str(parquet_path))
	cols = [c for c in pf.schema_arrow.names if c not in ("geo_accession", "__index_level_0__", "sample_id")]
	return len(cols)


def estimate_matrix_bytes(num_rows, num_genes, dtype_bytes=4):
	return int(num_rows) * int(num_genes) * int(dtype_bytes)


def _get_parquet_gene_layout(parquet_path):
	pf = pq.ParquetFile(str(parquet_path))
	cols = pf.schema_arrow.names
	gene_cols = [c for c in cols if c not in ("geo_accession", "__index_level_0__", "sample_id")]
	starts = [0]
	for rg in range(pf.metadata.num_row_groups):
		starts.append(starts[-1] + pf.metadata.row_group(rg).num_rows)
	return pf, gene_cols, starts


def load_selected_expression_rows(parquet_path, selected_rows):
	pf, gene_cols, row_group_starts = _get_parquet_gene_layout(parquet_path)
	x_out = np.empty((len(selected_rows), len(gene_cols)), dtype=np.float32)

	grouped = {}
	for out_i, global_row in enumerate(selected_rows):
		rg_idx = bisect.bisect_right(row_group_starts, global_row) - 1
		local_row = global_row - row_group_starts[rg_idx]
		grouped.setdefault(rg_idx, []).append((out_i, local_row))

	for rg_idx, reqs in grouped.items():
		local_rows = [row for _, row in reqs]
		rg_table = pf.read_row_group(rg_idx, columns=gene_cols, use_threads=True)
		sub = rg_table.take(pa.array(local_rows, type=pa.int64()))
		sub_np = SingleParquetStreamingMLMDataset._table_to_numpy(sub)
		for j, (out_i, _) in enumerate(reqs):
			x_out[out_i] = sub_np[j]

	return x_out


def format_float_for_tag(v):
	s = f"{v:.2e}" if (abs(v) < 1e-3 or abs(v) >= 1e3) else f"{v:.6f}"
	return s.replace(".", "p").replace("+", "").replace("-", "m")


def build_run_tag(cfg):
	return (
		f"norm-{cfg['normalization']}"
		f"_lr-{format_float_for_tag(cfg['learning_rate'])}"
		f"_wd-{format_float_for_tag(cfg['weight_decay'])}"
		f"_mask-{format_float_for_tag(cfg['mask_ratio'])}"
		f"_ree-{format_float_for_tag(cfg['ree_base'])}"
	)


def format_duration(seconds):
	seconds = max(0, int(seconds))
	hours = seconds // 3600
	minutes = (seconds % 3600) // 60
	secs = seconds % 60
	if hours > 0:
		return f"{hours:d}h {minutes:02d}m {secs:02d}s"
	if minutes > 0:
		return f"{minutes:d}m {secs:02d}s"
	return f"{secs:d}s"


def _init_ddp():
	local_rank = int(os.environ.get("LOCAL_RANK", 0))
	torch.cuda.set_device(local_rank)
	try:
		dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
	except TypeError:
		dist.init_process_group(backend="nccl")
	rank = dist.get_rank()
	world = dist.get_world_size()
	device = torch.device(f"cuda:{local_rank}")
	return rank, world, local_rank, device


def main():
	print("\n[STARTUP] train_single.py started - initializing DDP...", flush=True)
	script_start = time.time()

	rank, world_size, local_rank, device = _init_ddp()
	is_main = rank == 0

	if is_main:
		print("\n" + "=" * 70)
		print(f"ExpressionPerformer Single-Parquet Training - DDP ({world_size} processes)")
		print("=" * 70)
		print(f"\n[SETUP] Rank: {rank}, Device: {device}")

	if is_main and HAS_WANDB:
		wandb.init(project="Attention", config=CONFIG)
		for key in CONFIG:
			if key in wandb.config:
				CONFIG[key] = wandb.config[key]
		wandb.config.update(CONFIG, allow_val_change=True)

	cfg_obj = [CONFIG if is_main else None]
	dist.broadcast_object_list(cfg_obj, src=0)
	CONFIG.update(cfg_obj[0])
	# Always derive FFN width from hidden size for consistency in normal runs and sweeps.
	CONFIG["ffn_dim"] = int(CONFIG["hidden_dim"]) * 4
	if is_main and HAS_WANDB:
		wandb.config.update({"ffn_dim": CONFIG["ffn_dim"]}, allow_val_change=True)

	parquet_path = Path(CONFIG["expression_parquet"])
	if not parquet_path.exists():
		raise FileNotFoundError(f"Single parquet file not found: {parquet_path}")

	if is_main:
		print("\n[DATA] Building sample indices...", flush=True)
	t0 = time.time()

	train_rows = None
	val_rows = None
	if is_main:
		train_rows, val_rows = build_single_parquet_split(
			parquet_path=parquet_path,
			samples_json_path=CONFIG.get("samples_json", ""),
			train_subset=CONFIG.get("train_subset"),
			val_subset=CONFIG.get("val_subset"),
			balanced_sampling=CONFIG.get("balanced_sampling", True),
			seed=CONFIG.get("seed", 42),
			verbose=True,
		)

	train_rows_obj = [train_rows if is_main else None]
	val_rows_obj = [val_rows if is_main else None]
	dist.broadcast_object_list(train_rows_obj, src=0)
	dist.broadcast_object_list(val_rows_obj, src=0)
	train_rows = train_rows_obj[0]
	val_rows = val_rows_obj[0]

	if is_main:
		print(f"  + Index time: {time.time() - t0:.1f}s", flush=True)

	data_mode = str(CONFIG.get("data_mode", "streaming")).lower()
	if data_mode not in {"streaming", "preload"}:
		raise ValueError(f"Unsupported data_mode={data_mode!r}; expected 'streaming' or 'preload'.")

	num_genes = get_num_genes_from_single_parquet(parquet_path)
	if is_main:
		print(f"\n[CHECK] num_genes={num_genes}")
		assert num_genes > 10000, f"Expected ~16K genes, got {num_genes}"

	if data_mode == "preload":
		if is_main:
			train_bytes = estimate_matrix_bytes(len(train_rows), num_genes)
			val_bytes = estimate_matrix_bytes(len(val_rows), num_genes)
			total_gb = (train_bytes + val_bytes) / (1024 ** 3)
			print(f"[DATA] Preloading selected rows into RAM (~{total_gb:.2f} GiB raw float32).", flush=True)

		load_t0 = time.time()
		train_matrix = load_selected_expression_rows(parquet_path, train_rows)
		val_matrix = load_selected_expression_rows(parquet_path, val_rows)
		if is_main:
			print(f"[DATA] Preload complete in {time.time() - load_t0:.1f}s", flush=True)

		train_ds = InMemoryExpressionMLMDataset(
			train_matrix,
			normalization=CONFIG["normalization"],
			mask_ratio=CONFIG["mask_ratio"],
			mask_token=CONFIG["mask_token"],
		)
		val_ds = InMemoryExpressionMLMDataset(
			val_matrix,
			normalization=CONFIG["normalization"],
			mask_ratio=CONFIG["mask_ratio"],
			mask_token=CONFIG["mask_token"],
		)

		train_sampler = DistributedSampler(
			train_ds,
			num_replicas=world_size,
			rank=rank,
			shuffle=True,
			seed=CONFIG.get("seed", 42),
			drop_last=True,
		)
		val_sampler = DistributedSampler(
			val_ds,
			num_replicas=world_size,
			rank=rank,
			shuffle=False,
			drop_last=False,
		)
		train_batch_sampler = BatchSampler(train_sampler, batch_size=CONFIG["batch_size"], drop_last=True)
		val_batch_sampler = BatchSampler(val_sampler, batch_size=CONFIG["batch_size"], drop_last=False)
		train_epoch_controller = train_sampler
	else:
		train_ds = SingleParquetStreamingMLMDataset(
			parquet_path,
			train_rows,
			normalization=CONFIG["normalization"],
			mask_ratio=CONFIG["mask_ratio"],
			mask_token=CONFIG["mask_token"],
			cache_size=CONFIG.get("stream_cache_size", 2),
		)
		val_ds = SingleParquetStreamingMLMDataset(
			parquet_path,
			val_rows,
			normalization=CONFIG["normalization"],
			mask_ratio=CONFIG["mask_ratio"],
			mask_token=CONFIG["mask_token"],
			cache_size=CONFIG.get("stream_cache_size", 2),
		)

		train_batch_sampler = DistributedRowGroupBatchSampler(
			train_ds.group_to_indices,
			batch_size=CONFIG["batch_size"],
			num_replicas=world_size,
			rank=rank,
			shuffle=True,
			seed=CONFIG.get("seed", 42),
			drop_last=True,
		)
		val_batch_sampler = DistributedRowGroupBatchSampler(
			val_ds.group_to_indices,
			batch_size=CONFIG["batch_size"],
			num_replicas=world_size,
			rank=rank,
			shuffle=False,
			seed=CONFIG.get("seed", 42),
			drop_last=False,
		)
		train_epoch_controller = train_batch_sampler

	num_workers = int(CONFIG.get("num_workers", 0))
	if data_mode == "streaming" and num_workers > 0 and is_main:
		print("[DATA] Single-parquet streaming strongly prefers num_workers=0 for memory safety.", flush=True)

	loader_kwargs = {"num_workers": num_workers, "pin_memory": True}
	if num_workers > 0:
		loader_kwargs["prefetch_factor"] = int(CONFIG.get("prefetch_factor", 2))
		loader_kwargs["persistent_workers"] = bool(CONFIG.get("persistent_workers", False))

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

	if is_main:
		print(f"\n[DATA] Train: {len(train_ds):,} samples, {len(train_loader)} batches")
		print(f"[DATA] Val:   {len(val_ds):,} samples, {len(val_loader)} batches")

	dist.barrier()

	if is_main:
		print("\n[MODEL] Building ExpressionPerformer...")

	model = ExpressionPerformer(
		num_genes=num_genes,
		hidden_dim=CONFIG["hidden_dim"],
		n_heads=CONFIG["num_heads"],
		n_layers=CONFIG["num_layers"],
		ffn_dim=CONFIG["ffn_dim"],
		ree_base=CONFIG["ree_base"],
		mask_token_id=CONFIG["mask_token"],
		feature_type=CONFIG["feature_type"],
		compute_type=CONFIG["compute_type"],
		gradient_checkpointing=CONFIG.get("gradient_checkpointing", False),
	).to(device)

	model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
	total_params = sum(p.numel() for p in model.parameters())
	if is_main:
		print(f"  + Parameters: {total_params:,}")

	optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
	scaler = torch.amp.GradScaler("cuda", enabled=CONFIG.get("use_amp", True))

	if is_main:
		print(f"  + AdamW (lr={CONFIG['learning_rate']})")
		print("\n" + "=" * 70, flush=True)
		print("[TRAIN] Starting training...", flush=True)
		print("=" * 70 + "\n", flush=True)

	best_val_loss = float("inf")
	patience_counter = 0
	train_losses, val_losses = [], []

	ckpt_base = Path(CONFIG["checkpoint_dir"])
	run_timestamp = time.strftime("%Y%m%d_%H%M%S")
	run_id = wandb.run.id if (is_main and HAS_WANDB and wandb.run is not None) else run_timestamp
	run_tag = build_run_tag(CONFIG)
	ckpt_dir = ckpt_base / run_id
	if is_main:
		ckpt_base.mkdir(exist_ok=True, parents=True)
		ckpt_dir.mkdir(exist_ok=True, parents=True)

	global_best_path = ckpt_base / "global_best_val_loss.json"
	if global_best_path.exists():
		with open(global_best_path) as f:
			global_best_val_loss = json.load(f)["val_loss"]
	else:
		global_best_val_loss = float("inf")

	run_metadata = {
		"run_id": run_id,
		"timestamp": run_timestamp,
		"run_tag": run_tag,
		"normalization": CONFIG["normalization"],
		"sweep_parameters": {
			"learning_rate": CONFIG["learning_rate"],
			"weight_decay": CONFIG["weight_decay"],
			"mask_ratio": CONFIG["mask_ratio"],
			"ree_base": CONFIG["ree_base"],
			"early_stopping": CONFIG["early_stopping"],
		},
		"architecture": {
			"hidden_dim": CONFIG["hidden_dim"],
			"ffn_dim": CONFIG["ffn_dim"],
			"num_heads": CONFIG["num_heads"],
			"num_layers": CONFIG["num_layers"],
		},
		"dataset": {
			"train_samples_rank": len(train_ds),
			"val_samples_rank": len(val_ds),
			"num_genes": int(num_genes),
			"data_mode": data_mode,
			"balanced_sampling": CONFIG["balanced_sampling"],
			"train_subset": CONFIG["train_subset"],
			"val_subset": CONFIG["val_subset"],
			"expression_parquet": str(parquet_path),
		},
	}

	for epoch in range(CONFIG["epochs"]):
		epoch_start = time.time()
		train_epoch_controller.set_epoch(epoch)

		model.train()
		running_loss = 0.0
		num_batches = 0

		for batch_idx, (x_masked, x_true, mask_idx) in enumerate(train_loader):
			x_masked = x_masked.to(device, non_blocking=True)
			x_true = x_true.to(device, non_blocking=True)

			with torch.amp.autocast("cuda", enabled=CONFIG.get("use_amp", True)):
				pred = model(x_masked)

				loss_parts = []
				for i in range(len(x_masked)):
					idxs = mask_idx[i]
					if len(idxs) > 0:
						loss_parts.append(F.mse_loss(pred[i, idxs], x_true[i, idxs]))

				loss = torch.stack(loss_parts).mean() if loss_parts else torch.tensor(0.0, device=device)

			optimizer.zero_grad(set_to_none=True)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			running_loss += loss.item()
			num_batches += 1

			if is_main and (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
				avg = running_loss / num_batches
				elapsed_epoch = time.time() - epoch_start
				sec_per_batch = elapsed_epoch / max(1, num_batches)
				remaining_epoch_batches = max(0, len(train_loader) - (batch_idx + 1))
				epoch_eta_s = sec_per_batch * remaining_epoch_batches

				elapsed_total = time.time() - script_start
				done_total_batches = epoch * len(train_loader) + (batch_idx + 1)
				total_batches_run = max(1, CONFIG["epochs"] * len(train_loader))
				remaining_total_batches = max(0, total_batches_run - done_total_batches)
				total_eta_s = (elapsed_total / max(1, done_total_batches)) * remaining_total_batches
				print(
					f"  Epoch {epoch+1}/{CONFIG['epochs']} | "
					f"Batch {batch_idx+1}/{len(train_loader)} | "
					f"Loss: {loss.item():.6f} | Avg: {avg:.6f} | "
					f"{sec_per_batch:.3f}s/batch | "
					f"Epoch ETA: {format_duration(epoch_eta_s)} | "
					f"Run ETA: {format_duration(total_eta_s)}"
				)

		epoch_train_loss = running_loss / max(1, num_batches)

		model.eval()
		val_loss = 0.0
		val_batches = 0

		with torch.no_grad():
			for x_masked, x_true, mask_idx in val_loader:
				x_masked = x_masked.to(device, non_blocking=True)
				x_true = x_true.to(device, non_blocking=True)

				with torch.amp.autocast("cuda", enabled=CONFIG.get("use_amp", True)):
					pred = model(x_masked)

				loss_parts = []
				for i in range(len(x_masked)):
					idxs = mask_idx[i]
					if len(idxs) > 0:
						loss_parts.append(F.mse_loss(pred[i, idxs], x_true[i, idxs]))

				if loss_parts:
					val_loss += torch.stack(loss_parts).mean().item()
					val_batches += 1

		vl = torch.tensor(val_loss, device=device)
		vb = torch.tensor(float(val_batches), device=device)
		dist.all_reduce(vl, op=dist.ReduceOp.SUM)
		dist.all_reduce(vb, op=dist.ReduceOp.SUM)
		epoch_val_loss = (vl / vb.clamp(min=1)).item()

		train_losses.append(epoch_train_loss)
		val_losses.append(epoch_val_loss)
		scheduler.step()

		if is_main and HAS_WANDB:
			wandb.log(
				{
					"epoch": epoch + 1,
					"train_loss": epoch_train_loss,
					"val_loss": epoch_val_loss,
					"lr": scheduler.get_last_lr()[0],
				}
			)

		epoch_time = time.time() - epoch_start
		if is_main:
			model_sd = model.module.state_dict()
			print("\n  +==========================================+")
			print(f"  | Epoch {epoch+1}/{CONFIG['epochs']}")
			print(f"  | Train Loss: {epoch_train_loss:.6f}")
			print(f"  | Val Loss:   {epoch_val_loss:.6f}")
			print(f"  | Time: {epoch_time:.1f}s")

			payload = {
				"model_state_dict": model_sd,
				"optimizer_state_dict": optimizer.state_dict(),
				"scheduler_state_dict": scheduler.state_dict(),
				"epoch": epoch + 1,
				"train_loss": epoch_train_loss,
				"val_loss": epoch_val_loss,
				"config": dict(CONFIG),
				"run_metadata": run_metadata,
				"total_params": total_params,
			}

			torch.save(payload, ckpt_dir / f"epoch_{epoch:02d}.pt")

			if epoch_val_loss < best_val_loss:
				best_val_loss = epoch_val_loss
				patience_counter = 0
				torch.save(payload, ckpt_dir / "best_model.pt")
				torch.save(payload, ckpt_dir / f"best_{run_tag}_run-{run_id}.pt")
				print("  | + New best (run)! Saved best_model.pt")

				if epoch_val_loss < global_best_val_loss:
					global_best_val_loss = epoch_val_loss
					torch.save(payload, ckpt_base / "best_model.pt")
					with open(global_best_path, "w") as f:
						json.dump(
							{
								"val_loss": global_best_val_loss,
								"run_id": run_id,
								"epoch": epoch + 1,
								"run_tag": run_tag,
								"normalization": CONFIG["normalization"],
							},
							f,
							indent=2,
						)
					print(f"  | * New global best! {epoch_val_loss:.6f}")
			else:
				if CONFIG["early_stopping"]:
					patience_counter += 1
					print(f"  | - No improvement ({patience_counter}/{CONFIG['patience']})")
					if patience_counter >= CONFIG["patience"]:
						print("  | ! Early stopping!")
						print("  +==========================================+\n")
						break
				else:
					print("  | - No improvement (early_stopping=False; continuing)")

			print("  +==========================================+\n")

	if is_main:
		cfg = {
			**CONFIG,
			"num_genes": num_genes,
			"total_params": total_params,
			"best_val_loss": best_val_loss,
			"final_epoch": epoch + 1,
			"run_id": run_id,
			"timestamp": run_timestamp,
			"run_tag": run_tag,
			"dataset": run_metadata["dataset"],
			"architecture": run_metadata["architecture"],
			"sweep_parameters": run_metadata["sweep_parameters"],
		}
		with open(ckpt_dir / "config.json", "w") as f:
			json.dump(cfg, f, indent=2)

		with open(ckpt_dir / "run_metadata.json", "w") as f:
			json.dump(run_metadata, f, indent=2)

		pd.DataFrame(
			{"epoch": range(len(train_losses)), "train_loss": train_losses, "val_loss": val_losses}
		).to_csv(ckpt_dir / "loss_history.csv", index=False)

		if HAS_MATPLOTLIB:
			plt.figure(figsize=(10, 6))
			plt.plot(train_losses, marker="o", label="Train Loss", linewidth=2)
			plt.plot(val_losses, marker="s", label="Val Loss", linewidth=2)
			plt.xlabel("Epoch")
			plt.ylabel("MSE Loss")
			plt.title("ExpressionPerformer Single-Parquet Training")
			plt.legend()
			plt.grid(True, alpha=0.3)
			plt.tight_layout()
			plt.savefig(ckpt_dir / "loss_plot.png", dpi=150)
			plt.close()

		total_time = time.time() - script_start
		print("=" * 70)
		print(f"Training complete! {total_time:.0f}s ({total_time / 60:.1f}m)")
		print(f"  Run best val loss:    {best_val_loss:.6f}")
		print(f"  Global best val loss: {global_best_val_loss:.6f}")
		print(f"  Run checkpoints:      {ckpt_dir}/")
		print(f"  Global best model:    {ckpt_base / 'best_model.pt'}")
		print("=" * 70 + "\n")

	if is_main and HAS_WANDB:
		wandb.finish()

	dist.barrier()
	dist.destroy_process_group()


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"\n[ERROR] Exception in train_single.py: {e}", flush=True, file=sys.stderr)
		import traceback

		traceback.print_exc(file=sys.stderr)
		try:
			if dist.is_available() and dist.is_initialized():
				dist.destroy_process_group()
		except Exception:
			pass
		sys.exit(1)
