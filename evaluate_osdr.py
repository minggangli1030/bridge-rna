#!/usr/bin/env python3
# coding=utf-8
"""
Zero-shot OSDR evaluation for ExpressionPerformer checkpoints.

Downloads raw NASA OSDR mouse RNA-seq counts, preprocesses them to match
the bridge-rna training gene space, then evaluates each checkpoint using
masked-gene imputation at multiple mask ratios.

Usage:
  python evaluate_osdr.py \\
    --checkpoints checkpoints/human_5k/best_model.pt \\
                  checkpoints/mouse_5k/best_model.pt \\
                  checkpoints/mixed_5k/best_model.pt \\
    --output-dir results/osdr_eval

  # Skip download if cached parquet already exists:
  python evaluate_osdr.py \\
    --checkpoints checkpoints/human_5k/best_model.pt \\
    --osdr-parquet data/osdr/osdr_expression.parquet \\
    --output-dir results/osdr_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn.functional as F

from slim_performer_model import SLiMPerformerLayer

# ── Inline model definition (mirrors train_single.py) ─────────────────────────

class RotaryExpressionEmbedding(torch.nn.Module):
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


class ExpressionPerformer(torch.nn.Module):
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
        self.gene_embedding = torch.nn.Embedding(num_genes, hidden_dim)
        self.ree = RotaryExpressionEmbedding(hidden_dim, base=ree_base, mask_token_id=mask_token_id)
        self.layers = torch.nn.ModuleList(
            [
                SLiMPerformerLayer(hidden_dim, ffn_dim, n_heads, feature_type, compute_type, on_gptln=True)
                for _ in range(n_layers)
            ]
        )
        self.output_map = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, g = x.shape
        device = x.device
        gene_ids = torch.arange(g, device=device)
        gene_emb = self.gene_embedding(gene_ids)
        ree_emb = self.ree(x)
        h = gene_emb.unsqueeze(0) + ree_emb
        for layer in self.layers:
            rfs = layer.attention.sample_rfs(device)
            h = layer.full_forward(h, rfs)
        return self.output_map(h).squeeze(-1)


# ── Data paths ─────────────────────────────────────────────────────────────────

BRIDGE_RNA_DATA = Path(__file__).parent / "data"
ORTHOLOG_FILE = BRIDGE_RNA_DATA / "ensembl" / "orthologs_one2one.txt"
CANONICAL_GENES_FILE = BRIDGE_RNA_DATA / "ensembl" / "protein_coding_ortholog_genes.txt"
MOUSE_EXON_FILE = BRIDGE_RNA_DATA / "gencode" / "gencode_v49_mouse_gene_exon_lengths.csv"

# NASA OSDR metadata (included in bridge-rna repo via sp26_nasa reference)
# Users can override with --metadata-csv
DEFAULT_METADATA_CSV = Path(__file__).parent / "data" / "osdr" / "metadata_new.csv"
DEFAULT_OSDR_CACHE = Path(__file__).parent / "data" / "osdr"

# Mask ratios + block masking for evaluation
MASK_RATIOS = [0.15, 0.50, 0.80]
BLOCK_SIZES = [50]  # block masking: mask contiguous runs of this many genes
BATCH_SIZE = 16     # inference batch size


# ── Preprocessing helpers ──────────────────────────────────────────────────────

def load_ortholog_map() -> dict[str, str]:
    """Returns {mouse_gene_symbol → human_gene_symbol} for one-to-one orthologs."""
    df = pd.read_csv(ORTHOLOG_FILE, sep="\t")
    # columns: 'Gene name' (mouse), 'Human gene name' (human)
    df = df.dropna(subset=["Gene name", "Human gene name"])
    return dict(zip(df["Gene name"].str.strip(), df["Human gene name"].str.strip()))


def load_canonical_genes() -> list[str]:
    """Returns the ordered list of human gene symbols used in training."""
    df = pd.read_csv(CANONICAL_GENES_FILE, header=None)
    return [str(g).strip() for g in df.iloc[:, 0].tolist() if str(g).strip()]


def load_mouse_exon_lengths() -> pd.Series:
    """Returns Series indexed by mouse gene symbol with exon_length values."""
    df = pd.read_csv(MOUSE_EXON_FILE)
    # columns: gene_symbol, exon_length
    df = df.dropna(subset=["gene_symbol", "exon_length"])
    df = df.groupby("gene_symbol")["exon_length"].max()
    return df


def tpm_normalize_mouse(counts_df: pd.DataFrame, exon_lengths: pd.Series) -> pd.DataFrame:
    """
    TPM normalize mouse raw counts.
    counts_df: rows = genes (mouse symbols), cols = samples
    Returns TPM DataFrame, same shape.
    """
    L = exon_lengths.reindex(counts_df.index).fillna(1000).replace(0, 1000) / 1000.0
    rpk = counts_df.div(L, axis=0)
    scaling = rpk.sum(axis=0)
    tpm = rpk.div(scaling, axis=1) * 1e6
    return tpm.astype("float32")


def download_osdr_study(study_id: str, counts_filename: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    """
    Download a single OSDR study counts CSV from NASA GeneLab.
    Returns DataFrame indexed by gene, columns = samples, or None on failure.
    """
    import urllib.request

    # NASA GeneLab download URL pattern (as of 2025)
    # OSD-{id} = new naming, GLDS-{id} = legacy naming (still served)
    base_urls = [
        f"https://genelab-data.ndc.nasa.gov/genelab/static/media/dataset/{study_id}/rna_seq/{counts_filename}",
        f"https://osdr.nasa.gov/geneslab/static/media/dataset/{study_id}/rna_seq/{counts_filename}",
    ]

    out_path = cache_dir / "raw" / counts_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"    [cache hit] {counts_filename}", flush=True)
        return pd.read_csv(out_path, index_col=0)

    for url in base_urls:
        try:
            print(f"    Downloading {counts_filename} from {url} ...", flush=True)
            urllib.request.urlretrieve(url, out_path)
            df = pd.read_csv(out_path, index_col=0)
            print(f"    OK: {df.shape}", flush=True)
            return df
        except Exception as e:
            print(f"    WARN: {url} failed: {e}", flush=True)
            if out_path.exists():
                out_path.unlink()

    print(f"    SKIP: could not download {counts_filename}", flush=True)
    return None


def preprocess_osdr(
    metadata_csv: Path,
    osdr_raw_dir: Optional[Path],
    cache_dir: Path,
    ortholog_map: dict,
    canonical_genes: list[str],
    exon_lengths: pd.Series,
    download: bool = True,
) -> pd.DataFrame:
    """
    Build (or reload from cache) an OSDR expression matrix aligned to bridge-rna gene space.

    Returns DataFrame: rows = samples, cols = canonical_genes (human symbols), values = log1p(TPM).
    Missing genes filled with 0.
    """
    cache_parquet = cache_dir / "osdr_expression.parquet"
    if cache_parquet.exists():
        print(f"[OSDR] Loading cached parquet: {cache_parquet}", flush=True)
        return pd.read_parquet(cache_parquet)

    print(f"[OSDR] Building expression matrix from {metadata_csv} ...", flush=True)
    meta = pd.read_csv(metadata_csv)

    # Filter to mouse RNA-seq samples with spaceflight label
    required_cols = {"id.sample name", "id.accession", "counts_file", "counts_path",
                     "study.factor value.spaceflight", "study.characteristics.organism",
                     "has_rna_sequencing_rna_seq"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {missing}")

    meta = meta[meta["study.characteristics.organism"] == "Mus musculus"]
    meta = meta[meta["has_rna_sequencing_rna_seq"] == 1]
    meta = meta.dropna(subset=["counts_file", "id.accession"])
    print(f"[OSDR] {len(meta)} mouse RNA-seq samples across {meta['id.accession'].nunique()} studies", flush=True)

    # Build mouse gene → human gene map (symbol level)
    # ortholog_map: mouse_symbol → human_symbol
    canonical_set = set(canonical_genes)

    # Will accumulate per-study DataFrames (rows = samples, cols = mouse ENSMUSG or symbol)
    study_frames = []

    for study_id, grp in meta.groupby("id.accession"):
        counts_filename = grp["counts_file"].iloc[0]

        # Try to load from local raw dir first
        counts_df = None
        if osdr_raw_dir is not None:
            local_path = osdr_raw_dir / counts_filename
            if local_path.exists():
                print(f"  [{study_id}] Loading local {counts_filename}", flush=True)
                counts_df = pd.read_csv(local_path, index_col=0)

        if counts_df is None and download:
            counts_df = download_osdr_study(study_id, counts_filename, cache_dir)

        if counts_df is None:
            print(f"  [{study_id}] SKIPPED (no data)", flush=True)
            continue

        # counts_df: rows = genes, cols = sample names
        sample_names = grp["id.sample name"].tolist()
        available = [s for s in sample_names if s in counts_df.columns]
        if not available:
            # Try stripping whitespace
            counts_df.columns = counts_df.columns.str.strip()
            available = [s for s in sample_names if s in counts_df.columns]
        if not available:
            print(f"  [{study_id}] WARN: no matching sample columns found", flush=True)
            continue

        # Handle special case for GLDS-462
        if "GLDS-462" in counts_filename:
            counts_df.columns = counts_df.columns.str.replace("_mRNA", "", regex=False)
            available = [s for s in sample_names if s in counts_df.columns]

        sub = counts_df[available].copy()

        # QC: filter samples with too few nonzero genes
        nonzero = (sub > 0).sum(axis=0)
        sub = sub.loc[:, nonzero >= 8000]
        if sub.shape[1] == 0:
            print(f"  [{study_id}] WARN: all samples failed QC", flush=True)
            continue

        # TPM normalize
        tpm = tpm_normalize_mouse(sub, exon_lengths)

        # Map gene index to human symbols
        # counts_df index could be mouse symbols (Akt1, Trp53...) OR ENSMUSG IDs
        idx = tpm.index
        if idx[0].startswith("ENSMUSG"):
            # Need ENSMUSG → mouse symbol → human symbol
            # Use the sp26_nasa osdr orthologs for ENSMUSG mapping
            ensmusg_to_human = _build_ensmusg_to_human_map()
            tpm.index = [ensmusg_to_human.get(g, "") for g in idx]
        else:
            # mouse symbol → human symbol
            tpm.index = [ortholog_map.get(g, ortholog_map.get(g.capitalize(), "")) for g in idx]

        tpm = tpm[tpm.index != ""]  # drop unmapped
        tpm = tpm[~tpm.index.duplicated(keep="first")]

        # Keep only canonical genes
        keep = tpm.index.isin(canonical_set)
        tpm = tpm[keep]
        if tpm.shape[0] == 0:
            print(f"  [{study_id}] WARN: no canonical genes after mapping", flush=True)
            continue

        # log1p transform; transpose to (samples x genes)
        log_tpm = np.log1p(tpm).T.astype("float32")

        # Attach spaceflight label
        label_map = dict(zip(grp["id.sample name"], grp["study.factor value.spaceflight"].notna().astype(int)))
        log_tpm["spaceflight"] = log_tpm.index.map(lambda s: 1 if str(label_map.get(s, "")).lower() == "space flight" else 0)
        log_tpm["study_id"] = study_id
        study_frames.append(log_tpm)
        print(f"  [{study_id}] {log_tpm.shape[0]} samples, {(keep).sum()} genes matched", flush=True)

    if not study_frames:
        raise RuntimeError("No OSDR studies loaded — check metadata, raw data dir, and download flag.")

    expr = pd.concat(study_frames, axis=0)

    # Align to canonical gene order; fill missing with 0
    meta_cols = ["spaceflight", "study_id"]
    gene_cols = [g for g in canonical_genes if g in expr.columns]
    missing_genes = [g for g in canonical_genes if g not in expr.columns]
    print(f"[OSDR] {len(gene_cols)}/{len(canonical_genes)} canonical genes present, {len(missing_genes)} filled with 0", flush=True)

    aligned = expr[gene_cols].reindex(columns=canonical_genes, fill_value=0.0).astype("float32")
    aligned[meta_cols] = expr[meta_cols].values
    aligned.index.name = "sample_id"

    cache_dir.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(cache_parquet)
    print(f"[OSDR] Cached to {cache_parquet}  shape={aligned.shape}", flush=True)
    return aligned


def _build_ensmusg_to_human_map() -> dict[str, str]:
    """
    Build ENSMUSG → human gene symbol map from bridge-rna reference data.
    Requires the sp26_nasa orthologs CSV at the default path, or falls back
    to the bridge-rna orthologs.
    """
    # Try sp26_nasa OSDR orthologs first (has ENSMUSG IDs)
    candidates = [
        Path(__file__).parent / "data" / "osdr" / "human_mouse_orthologs.csv",
        Path(__file__).parent.parent / "sp26_nasa" / "osdr_data" / "human_mouse_orthologs.csv",
        Path(__file__).parent.parent / "workspace" / "sp26_nasa" / "osdr_data" / "human_mouse_orthologs.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df = df[df.get("Mouse homology type", "ortholog_one2one") == "ortholog_one2one"]
            return dict(zip(df["Mouse gene stable ID"].str.strip(), df["Gene name"].str.strip()))

    return {}


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[ExpressionPerformer, dict, list[str]]:
    """
    Load ExpressionPerformer from a checkpoint saved by train_single.py.
    Returns (model, config, gene_list) where gene_list is ordered gene symbols.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}))
    sd = {k.replace("module.", "", 1): v for k, v in ckpt["model_state_dict"].items()}

    num_genes = int(cfg.get("num_genes") or sd["gene_embedding.weight"].shape[0])

    model = ExpressionPerformer(
        num_genes=num_genes,
        hidden_dim=int(cfg.get("hidden_dim", 768)),
        n_heads=int(cfg.get("num_heads", 8)),
        n_layers=int(cfg.get("num_layers", 2)),
        ffn_dim=int(cfg.get("ffn_dim", cfg.get("hidden_dim", 768) * 4)),
        ree_base=float(cfg.get("ree_base", 100.0)),
        mask_token_id=int(cfg.get("mask_token", -10)),
        feature_type=str(cfg.get("feature_type", "sqr")),
        compute_type=str(cfg.get("compute_type", "iter")),
    )
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    # Recover gene list from training parquet if path stored in config
    gene_list = []
    parquet_path = cfg.get("expression_parquet", "")
    if parquet_path and Path(parquet_path).exists():
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(parquet_path)
        gene_list = [c for c in pf.schema_arrow.names
                     if c not in ("geo_accession", "__index_level_0__", "sample_id")]

    return model, cfg, gene_list


# ── Evaluation ─────────────────────────────────────────────────────────────────

def pearson_per_sample(pred: np.ndarray, true: np.ndarray, mask_idx: np.ndarray) -> np.ndarray:
    """Per-sample Pearson r, computed only on masked positions."""
    rs = []
    for i in range(len(pred)):
        p = pred[i, mask_idx[i]]
        t = true[i, mask_idx[i]]
        if len(p) < 3 or np.std(p) < 1e-9 or np.std(t) < 1e-9:
            rs.append(float("nan"))
        else:
            rs.append(float(np.corrcoef(p, t)[0, 1]))
    return np.array(rs)


def spearman_per_sample(pred: np.ndarray, true: np.ndarray, mask_idx: np.ndarray) -> np.ndarray:
    rs = []
    for i in range(len(pred)):
        p = pred[i, mask_idx[i]]
        t = true[i, mask_idx[i]]
        if len(p) < 3 or np.std(p) < 1e-9 or np.std(t) < 1e-9:
            rs.append(float("nan"))
        else:
            r, _ = stats.spearmanr(p, t)
            rs.append(float(r))
    return np.array(rs)


def mse_per_sample(pred: np.ndarray, true: np.ndarray, mask_idx: np.ndarray) -> np.ndarray:
    mses = []
    for i in range(len(pred)):
        p = pred[i, mask_idx[i]]
        t = true[i, mask_idx[i]]
        mses.append(float(np.mean((p - t) ** 2)))
    return np.array(mses)


def make_mask_random(num_genes: int, num_mask: int, n: int, rng: np.random.Generator) -> np.ndarray:
    mask_idx = np.stack([rng.choice(num_genes, num_mask, replace=False) for _ in range(n)])
    return mask_idx


def make_mask_block(num_genes: int, block_size: int, n: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, num_genes - block_size, size=n)
    mask_idx = np.stack([np.arange(s, s + block_size) for s in starts])
    return mask_idx


def evaluate_model(
    model: ExpressionPerformer,
    x_true: np.ndarray,   # (N, num_train_genes) — aligned to training gene space
    osdr_gene_mask: np.ndarray,  # bool (num_train_genes,) — True where OSDR has data
    mask_ratios: list[float],
    block_sizes: list[int],
    batch_size: int,
    device: torch.device,
    mask_token: float = -10.0,
    rng_seed: int = 42,
) -> dict:
    """
    Runs zero-shot imputation on x_true.
    x_true: log1p(TPM) in training gene order (missing genes already zeroed).
    Returns dict of metrics per masking strategy.
    """
    rng = np.random.default_rng(rng_seed)
    N, G = x_true.shape
    results = {}

    # Pre-compute baselines using available genes only
    avail = x_true[:, osdr_gene_mask]  # (N, num_osdr_genes)
    gene_mean = avail.mean(axis=0)  # (num_osdr_genes,)
    sample_mean = avail.mean(axis=1, keepdims=True)  # (N, 1)

    strategies = (
        [(f"random_{int(r*100)}pct", "random", int(G * r)) for r in mask_ratios]
        + [(f"block_{b}", "block", b) for b in block_sizes]
    )

    for strat_name, strat_type, param in strategies:
        print(f"    masking: {strat_name} ...", flush=True)
        all_pearson, all_spearman, all_mse = [], [], []
        all_pearson_gene_mean, all_pearson_sample_mean = [], []
        all_mse_gene_mean, all_mse_sample_mean = [], []

        for batch_start in range(0, N, batch_size):
            batch = x_true[batch_start : batch_start + batch_size]
            b = len(batch)

            if strat_type == "random":
                mask_idx = make_mask_random(G, param, b, rng)
            else:
                mask_idx = make_mask_block(G, param, b, rng)

            # Build masked input
            x_masked = batch.copy()
            for i in range(b):
                x_masked[i, mask_idx[i]] = mask_token

            # Model prediction
            with torch.no_grad():
                x_t = torch.from_numpy(x_masked).to(device)
                pred_t = model(x_t).cpu().numpy()

            # Metrics (model)
            all_pearson.append(pearson_per_sample(pred_t, batch, mask_idx))
            all_spearman.append(spearman_per_sample(pred_t, batch, mask_idx))
            all_mse.append(mse_per_sample(pred_t, batch, mask_idx))

            # Baselines — for masked positions, predict gene mean or sample mean
            # Gene mean baseline: for each masked gene j, predict gene_mean[j]
            gene_mean_full = np.zeros(G, dtype=np.float32)
            gene_mean_full[osdr_gene_mask] = gene_mean
            pred_gm = np.broadcast_to(gene_mean_full, (b, G)).copy()
            all_pearson_gene_mean.append(pearson_per_sample(pred_gm, batch, mask_idx))
            all_mse_gene_mean.append(mse_per_sample(pred_gm, batch, mask_idx))

            # Sample mean baseline: predict each sample's own mean for masked positions
            pred_sm = np.broadcast_to(sample_mean[batch_start : batch_start + b], (b, G)).copy()
            all_pearson_sample_mean.append(pearson_per_sample(pred_sm, batch, mask_idx))
            all_mse_sample_mean.append(mse_per_sample(pred_sm, batch, mask_idx))

        def cat(lst):
            return np.concatenate(lst)

        def nanmean(arr):
            return float(np.nanmean(arr))

        def nanmedian(arr):
            return float(np.nanmedian(arr))

        results[strat_name] = {
            "n_samples": N,
            "pearson_mean":        nanmean(cat(all_pearson)),
            "pearson_median":      nanmedian(cat(all_pearson)),
            "spearman_mean":       nanmean(cat(all_spearman)),
            "spearman_median":     nanmedian(cat(all_spearman)),
            "mse_mean":            nanmean(cat(all_mse)),
            "baseline_gene_mean_pearson":   nanmean(cat(all_pearson_gene_mean)),
            "baseline_sample_mean_pearson": nanmean(cat(all_pearson_sample_mean)),
            "baseline_gene_mean_mse":       nanmean(cat(all_mse_gene_mean)),
            "baseline_sample_mean_mse":     nanmean(cat(all_mse_sample_mean)),
        }

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zero-shot OSDR evaluation")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to best_model.pt checkpoints (one per variant)")
    parser.add_argument("--output-dir", default="results/osdr_eval",
                        help="Directory to write results CSV and JSON")
    parser.add_argument("--osdr-parquet", default=None,
                        help="Pre-built OSDR expression parquet (skips download+preprocess)")
    parser.add_argument("--metadata-csv", default=None,
                        help="OSDR metadata CSV (default: data/osdr/metadata_new.csv)")
    parser.add_argument("--osdr-raw-dir", default=None,
                        help="Directory containing pre-downloaded GLDS-*.csv raw count files")
    parser.add_argument("--no-download", action="store_true",
                        help="Do not download OSDR data from NASA (requires --osdr-parquet or --osdr-raw-dir)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="auto",
                        help="cuda / cpu / auto")
    parser.add_argument("--wandb", action="store_true", help="Log results to W&B")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[EVAL] Device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load gene reference data ────────────────────────────────────────────────
    print("[EVAL] Loading reference gene data...", flush=True)
    ortholog_map = load_ortholog_map()
    canonical_genes = load_canonical_genes()
    exon_lengths = load_mouse_exon_lengths()
    print(f"  + {len(ortholog_map):,} mouse→human ortholog pairs")
    print(f"  + {len(canonical_genes):,} canonical genes")

    # ── Build / load OSDR expression matrix ────────────────────────────────────
    if args.osdr_parquet:
        osdr_parquet = Path(args.osdr_parquet)
        if not osdr_parquet.exists():
            raise FileNotFoundError(f"OSDR parquet not found: {osdr_parquet}")
        print(f"[EVAL] Loading OSDR parquet: {osdr_parquet}", flush=True)
        osdr_df = pd.read_parquet(osdr_parquet)
    else:
        metadata_csv = Path(args.metadata_csv) if args.metadata_csv else DEFAULT_METADATA_CSV
        if not metadata_csv.exists():
            # Try to copy from sp26_nasa
            sp26 = Path(__file__).parent.parent / "workspace" / "sp26_nasa" / "osdr_data" / "metadata_new.csv"
            if sp26.exists():
                import shutil
                metadata_csv.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(sp26, metadata_csv)
                # Also copy orthologs
                orth_src = sp26.parent / "human_mouse_orthologs.csv"
                if orth_src.exists():
                    shutil.copy(orth_src, metadata_csv.parent / "human_mouse_orthologs.csv")
                print(f"[EVAL] Copied OSDR metadata from sp26_nasa → {metadata_csv}", flush=True)
            else:
                raise FileNotFoundError(
                    f"OSDR metadata not found at {metadata_csv}.\n"
                    "Supply --metadata-csv or ensure data/osdr/metadata_new.csv exists."
                )

        raw_dir = Path(args.osdr_raw_dir) if args.osdr_raw_dir else None
        osdr_df = preprocess_osdr(
            metadata_csv=metadata_csv,
            osdr_raw_dir=raw_dir,
            cache_dir=DEFAULT_OSDR_CACHE,
            ortholog_map=ortholog_map,
            canonical_genes=canonical_genes,
            exon_lengths=exon_lengths,
            download=not args.no_download,
        )

    print(f"[EVAL] OSDR shape: {osdr_df.shape}", flush=True)

    # Separate expression from metadata columns
    meta_cols = [c for c in osdr_df.columns if c in ("spaceflight", "study_id")]
    gene_cols = [c for c in canonical_genes if c in osdr_df.columns]
    x_osdr = osdr_df[canonical_genes].values.astype("float32")  # (N, num_canonical_genes)
    osdr_gene_mask = np.array([(g in set(osdr_df.columns)) for g in canonical_genes], dtype=bool)

    spaceflight_labels = osdr_df.get("spaceflight", pd.Series(np.zeros(len(osdr_df), dtype=int))).values

    print(f"[EVAL] {x_osdr.shape[0]} samples: {spaceflight_labels.sum()} spaceflight, "
          f"{(spaceflight_labels == 0).sum()} control", flush=True)
    print(f"[EVAL] Gene coverage: {osdr_gene_mask.sum()}/{len(canonical_genes)} canonical genes present in OSDR", flush=True)

    # ── W&B ────────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        try:
            import wandb as wb
            wandb_run = wb.init(
                project=os.environ.get("WANDB_PROJECT", "bridge-rna"),
                name="osdr_eval",
                group="osdr_benchmark",
                config={
                    "n_samples": x_osdr.shape[0],
                    "n_genes": x_osdr.shape[1],
                    "osdr_gene_coverage": int(osdr_gene_mask.sum()),
                    "mask_ratios": MASK_RATIOS,
                    "block_sizes": BLOCK_SIZES,
                    "spaceflight_samples": int(spaceflight_labels.sum()),
                    "control_samples": int((spaceflight_labels == 0).sum()),
                },
            )
        except Exception as e:
            print(f"[WARN] W&B init failed: {e}", flush=True)

    # ── Evaluate each checkpoint ───────────────────────────────────────────────
    all_results = []

    for ckpt_path_str in args.checkpoints:
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.exists():
            print(f"[WARN] Checkpoint not found: {ckpt_path} — skipping", flush=True)
            continue

        # Infer variant name from path
        variant = ckpt_path.parent.name
        if variant == ckpt_path.stem:
            variant = ckpt_path.parent.parent.name
        print(f"\n[EVAL] === Evaluating {variant} ({ckpt_path}) ===", flush=True)

        t0 = time.time()
        model, cfg, train_gene_list = load_checkpoint(ckpt_path, device)

        # If we have training gene list, verify alignment
        if train_gene_list:
            if train_gene_list != canonical_genes:
                print(f"  [WARN] Training gene list ({len(train_gene_list)}) differs from canonical ({len(canonical_genes)})")
                # Re-align OSDR to training gene order
                gene_order = train_gene_list
                x_aligned = pd.DataFrame(x_osdr, columns=canonical_genes).reindex(columns=gene_order, fill_value=0.0).values.astype("float32")
                gene_mask_aligned = np.array([(g in set(gene_cols)) for g in gene_order], dtype=bool)
            else:
                x_aligned = x_osdr
                gene_mask_aligned = osdr_gene_mask
        else:
            x_aligned = x_osdr
            gene_mask_aligned = osdr_gene_mask

        print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)
        print(f"  Evaluating {x_aligned.shape[0]} samples, {x_aligned.shape[1]} genes ...", flush=True)

        mask_token = float(cfg.get("mask_token", -10.0))

        metrics = evaluate_model(
            model=model,
            x_true=x_aligned,
            osdr_gene_mask=gene_mask_aligned,
            mask_ratios=MASK_RATIOS,
            block_sizes=BLOCK_SIZES,
            batch_size=args.batch_size,
            device=device,
            mask_token=mask_token,
        )

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s", flush=True)

        # Pretty-print summary
        print(f"\n  {'Masking':<20} {'Pearson':>10} {'Spearman':>10} {'MSE':>10} {'BL_GeneMean':>12} {'BL_SampMean':>12}")
        print(f"  {'-'*76}")
        for strat, m in metrics.items():
            print(f"  {strat:<20} {m['pearson_mean']:>10.4f} {m['spearman_mean']:>10.4f} "
                  f"{m['mse_mean']:>10.4f} {m['baseline_gene_mean_pearson']:>12.4f} "
                  f"{m['baseline_sample_mean_pearson']:>12.4f}")

        for strat, m in metrics.items():
            row = {
                "variant": variant,
                "checkpoint": str(ckpt_path),
                "masking_strategy": strat,
                **m,
            }
            all_results.append(row)

        # W&B logging
        if wandb_run is not None:
            for strat, m in metrics.items():
                wb.log({f"{variant}/{strat}/{k}": v for k, v in m.items()})

    # ── Save results ───────────────────────────────────────────────────────────
    if not all_results:
        print("\n[WARN] No results to save.", flush=True)
        return

    results_df = pd.DataFrame(all_results)
    results_csv = output_dir / "osdr_eval_results.csv"
    results_json = output_dir / "osdr_eval_results.json"
    results_df.to_csv(results_csv, index=False)
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[EVAL] Results saved:")
    print(f"  CSV:  {results_csv}")
    print(f"  JSON: {results_json}")

    # Summary table
    print("\n[EVAL] === SUMMARY: Pearson correlation by variant and masking strategy ===")
    pivot = results_df.pivot_table(
        index="variant", columns="masking_strategy", values="pearson_mean", aggfunc="first"
    )
    print(pivot.to_string())

    if wandb_run is not None:
        wandb_run.log({"summary_table": wb.Table(dataframe=results_df)})
        wandb_run.finish()


if __name__ == "__main__":
    main()
