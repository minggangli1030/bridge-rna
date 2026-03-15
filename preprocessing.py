"""
RNA-seq preprocessing module for ARCHS4 expression data.

Handles: ortholog loading, gene selection, TPM normalization,
log transformation, QC filtering, cross-split deduplication, and export.

Usage:
    from preprocessing import RNADatasetBuilder

    builder = RNADatasetBuilder(
        species="both",
        gene_set="shared_orthologs",
    )
    builder.process()
    builder.save_parquet("output/")

    # Split later with scikit-learn
    X, meta = builder.get_data()
"""

import os
import gc
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class PreprocessingConfig:
    """All preprocessing parameters in one place."""

    # --- Species & gene set ---
    species: str = "both"                    # "human", "mouse", "both"
    gene_set: str = "shared_orthologs"       # "shared_orthologs", "union_orthologs"

    # --- QC ---
    qc_min_nonzero: int = 14_000             # min non-zero genes per sample
    remove_single_cell: bool = True

    # --- Normalization ---
    normalization: str = "tpm"               # "log1p_tpm", "tpm", "raw_counts"
    extraction_batch_size: int = 10_000

    # --- Subsetting ---
    max_samples_per_species: Optional[int] = None  # None = all data

    # --- Paths ---
    archs4_dir: str = "data/archs4"
    orthologs_file: str = "data/ensembl/orthologs_one2one.txt"
    protein_coding_file: str = "data/ensembl/protein_coding_ortholog_genes.txt"
    exon_lengths_human: str = "data/gencode/gencode_v49_gene_exon_lengths.csv"
    exon_lengths_mouse: str = "data/gencode/gencode_v49_mouse_gene_exon_lengths.csv"
    output_dir: str = "data/archs4/train_orthologs"

    # --- Reproducibility ---
    seed: int = 42


# ============================================================
# GENE REGISTRY
# ============================================================
class GeneRegistry:
    """
    Loads ortholog mappings and determines the canonical gene list
    based on the chosen gene_set mode.
    """

    def __init__(self, orthologs_file: str, protein_coding_file: str):
        self.ortho_df = pd.read_csv(orthologs_file, sep="\t")

        # Load protein-coding gene whitelist (human symbols)
        with open(protein_coding_file) as f:
            self.protein_coding = set(line.strip() for line in f if line.strip())

        # Mouse→Human and Human→Mouse mappings
        self.mouse_to_human = dict(
            zip(self.ortho_df["Gene name"], self.ortho_df["Human gene name"])
        )
        self.human_to_mouse = dict(
            zip(self.ortho_df["Human gene name"], self.ortho_df["Gene name"])
        )

        # All valid human ortholog gene names that are protein-coding
        self.all_human_ortho = sorted(
            g for g in self.ortho_df["Human gene name"].unique()
            if isinstance(g, str) and g in self.protein_coding
        )
        self.all_mouse_ortho = sorted(
            g for g in self.ortho_df["Gene name"].unique()
            if isinstance(g, str) and self.mouse_to_human.get(g) in self.protein_coding
        )

    def get_canonical_genes(
        self,
        gene_set: str,
        human_zero_genes: Optional[set] = None,
        mouse_zero_genes: Optional[set] = None,
    ) -> list[str]:
        """
        Return the canonical gene list (human gene symbols, sorted).

        Args:
            gene_set: "shared_orthologs" or "union_orthologs"
            human_zero_genes: genes with zero expression across all human samples
            mouse_zero_genes: genes with zero expression across all mouse samples
        """
        human_zero = human_zero_genes or set()
        mouse_zero = mouse_zero_genes or set()

        if gene_set == "shared_orthologs":
            # Only genes expressed in BOTH species
            genes = [
                g for g in self.all_human_ortho
                if g not in human_zero and g not in mouse_zero
            ]
        elif gene_set == "union_orthologs":
            # All ortholog genes, even if expressed in only one species
            genes = list(self.all_human_ortho)
        else:
            raise ValueError(f"Unknown gene_set: {gene_set!r}")

        return sorted(genes)


# ============================================================
# EXPRESSION LOADER
# ============================================================
class ExpressionLoader:
    """Loads and normalizes expression data from ARCHS4 H5 files via h5py."""

    SC_THRESHOLD = 0.5  # single-cell probability cutoff

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._exon_human = None
        self._exon_mouse = None

    @property
    def exon_lengths_human(self) -> pd.Series:
        if self._exon_human is None:
            df = pd.read_csv(self.config.exon_lengths_human)
            self._exon_human = df.set_index("gene_symbol")["exon_length"]
        return self._exon_human

    @property
    def exon_lengths_mouse(self) -> pd.Series:
        if self._exon_mouse is None:
            df = pd.read_csv(self.config.exon_lengths_mouse)
            self._exon_mouse = df.set_index("gene_symbol")["exon_length"]
        return self._exon_mouse

    def _load_h5_meta(self, h5_path: str, species: str):
        """Read H5 metadata: gene symbols, GEO accessions, SC probabilities."""
        import h5py
        print(f"    Opening {h5_path}...")
        h = h5py.File(h5_path, "r")

        n_genes, n_samples = h["data/expression"].shape
        print(f"    H5 shape: {n_genes:,} genes × {n_samples:,} samples")

        print(f"    Loading gene symbols...", end=" ", flush=True)
        gene_symbols = np.array([g.decode() for g in h["meta/genes/symbol"][:]])
        print(f"done ({len(gene_symbols):,})")

        print(f"    Loading sample accessions...", end=" ", flush=True)
        geo_accessions = np.array([s.decode() for s in h["meta/samples/geo_accession"][:]])
        print(f"done ({len(geo_accessions):,})")

        sc_prob = None
        if self.config.remove_single_cell and "singlecellprobability" in h["meta/samples"]:
            print(f"    Loading SC probabilities...", end=" ", flush=True)
            sc_prob = h["meta/samples/singlecellprobability"][:]
            n_bulk = int((sc_prob < self.SC_THRESHOLD).sum())
            print(f"done ({n_bulk:,} bulk / {len(sc_prob) - n_bulk:,} SC)")

        gene_lengths = (
            self.exon_lengths_human if species == "human"
            else self.exon_lengths_mouse
        )
        print(f"    Computing gene mask...", end=" ", flush=True)
        valid_genes = set(gene_lengths.index)
        gene_mask = np.array([g in valid_genes for g in gene_symbols], dtype=bool)
        print(f"done — {gene_mask.sum():,} / {len(gene_symbols):,} genes with exon lengths")

        return h, gene_symbols, geo_accessions, sc_prob, gene_mask, gene_lengths

    def _normalize_df(
        self,
        chunk_df: pd.DataFrame,
        gene_lengths: pd.Series,
        canonical_genes: list[str],
        gene_name_map: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Shared normalization: groupby → QC → TPM → remap → reindex.
        Normalization uses original gene names so gene_lengths match,
        then remaps (e.g. mouse→human) afterward.
        Returns aligned DataFrame or empty DataFrame if nothing passes QC.
        """
        cfg = self.config

        # Aggregate duplicate gene rows
        chunk_df = chunk_df.groupby(level=0).sum()

        # QC: min non-zero genes (before remap — gene count is the same)
        nonzero = (chunk_df > 0).sum(axis=0)
        chunk_df = chunk_df[nonzero[nonzero >= cfg.qc_min_nonzero].index]

        if chunk_df.shape[1] == 0:
            return chunk_df

        # Normalize (using original species gene names that match gene_lengths)
        if cfg.normalization in ("log1p_tpm", "tpm"):
            lengths_kb = gene_lengths.reindex(chunk_df.index).fillna(1000) / 1000.0
            rate = chunk_df.div(lengths_kb, axis=0)
            tpm = rate.div(rate.sum(axis=0), axis=1) * 1e6
            chunk_df = np.log1p(tpm) if cfg.normalization == "log1p_tpm" else tpm

        # Remap gene names (e.g. mouse → human symbols) AFTER normalization
        if gene_name_map is not None:
            new_idx = [gene_name_map.get(g, g) for g in chunk_df.index]
            chunk_df.index = new_idx
            chunk_df = chunk_df.groupby(level=0).sum()

        # Align to canonical gene list
        chunk_df = chunk_df.reindex(canonical_genes, fill_value=0).astype("float32")
        return chunk_df

    def _extract_subset(
        self,
        h5_path: str,
        species: str,
        canonical_genes: list[str],
        tmp_dir: str,
        gene_name_map: Optional[dict] = None,
    ) -> tuple[list[Path], pd.DataFrame]:
        """
        Fast subset extraction using archs4py.
        Returns (batch_paths, metadata_df).
        """
        import archs4py as a4

        cfg = self.config
        n_samples = cfg.max_samples_per_species
        t0 = time.time()

        gene_lengths = (
            self.exon_lengths_human if species == "human"
            else self.exon_lengths_mouse
        )

        print(f"    SUBSET MODE via archs4py: {n_samples:,} random samples (seed={cfg.seed})")
        print(f"    Reading from {h5_path}...", flush=True)
        raw_df = a4.data.rand(h5_path, n_samples, seed=cfg.seed, remove_sc=True)
        print(f"    Got {raw_df.shape[1]:,} bulk samples × {raw_df.shape[0]:,} genes "
              f"in {time.time() - t0:.0f}s")

        # Filter to genes with exon lengths
        valid_genes = set(gene_lengths.index)
        keep = [g for g in raw_df.index if g in valid_genes]
        raw_df = raw_df.loc[keep]
        print(f"    Filtered to {len(keep):,} genes with exon lengths")

        # Normalize
        chunk_df = self._normalize_df(raw_df, gene_lengths, canonical_genes, gene_name_map)
        del raw_df

        if chunk_df.shape[1] == 0:
            print(f"    0 samples passed QC")
            return [], pd.DataFrame(columns=["geo_accession", "species"])

        print(f"    {chunk_df.shape[1]:,} samples passed QC")

        # Write single batch parquet
        tmp = Path(tmp_dir)
        tmp.mkdir(parents=True, exist_ok=True)
        batch_path = tmp / f"{species}_batch_0001.parquet"
        chunk_df.to_parquet(batch_path, compression="zstd")

        metadata = pd.DataFrame({
            "geo_accession": chunk_df.columns.tolist(),
            "species": species,
        })
        print(f"    Species done in {time.time() - t0:.0f}s — "
              f"{chunk_df.shape[1]:,} samples")
        return [batch_path], metadata

    def extract_and_normalize(
        self,
        h5_path: str,
        species: str,
        canonical_genes: list[str],
        tmp_dir: str,
        gene_name_map: Optional[dict] = None,
    ) -> tuple[list[Path], pd.DataFrame]:
        """
        Read all bulk samples from ARCHS4 H5, normalize, align.
        Writes each batch to a temporary parquet file to avoid OOM.

        For subset mode, delegates to _extract_subset (archs4py).

        Args:
            gene_name_map: Optional dict mapping H5 gene symbols to canonical names
                           (e.g. mouse→human for ortholog alignment).

        Returns:
            (batch_paths, metadata_df)
        """
        # Fast path for subset mode
        if self.config.max_samples_per_species is not None:
            return self._extract_subset(
                h5_path, species, canonical_genes, tmp_dir, gene_name_map
            )

        h, gene_symbols, geo_accessions, sc_prob, gene_mask, gene_lengths = \
            self._load_h5_meta(h5_path, species)

        cfg = self.config
        expr_ds = h["data/expression"]  # (n_genes, n_samples)
        n_total = expr_ds.shape[1]
        batch_size = cfg.extraction_batch_size
        gene_names = gene_symbols[gene_mask]

        batch_paths = []
        all_accessions = []
        total = 0
        total_bulk = 0
        t_species = time.time()

        n_batches = (n_total + batch_size - 1) // batch_size
        print(f"    Starting extraction: {n_batches} batches of {batch_size:,}")

        tmp = Path(tmp_dir)
        tmp.mkdir(parents=True, exist_ok=True)

        for batch_num in range(1, n_batches + 1):
            b_start = (batch_num - 1) * batch_size
            b_end = min(batch_num * batch_size, n_total)

            print(f"    Batch {batch_num}/{n_batches}: reading {b_end - b_start:,} samples from H5...",
                  end=" ", flush=True)
            try:
                raw = expr_ds[:, b_start:b_end]  # (n_genes, chunk_size)
            except OSError as e:
                print(f"SKIPPED (H5 read error: {e})")
                continue

            # Filter to genes with exon lengths (in memory)
            raw = raw[gene_mask]

            # Filter single-cell samples
            if sc_prob is not None:
                bulk_mask = sc_prob[b_start:b_end] < self.SC_THRESHOLD
                raw = raw[:, bulk_mask]
                chunk_accessions = geo_accessions[b_start:b_end][bulk_mask]
            else:
                chunk_accessions = geo_accessions[b_start:b_end]

            if raw.shape[1] == 0:
                print("skipped (all SC)")
                continue

            total_bulk += raw.shape[1]

            chunk_df = pd.DataFrame(raw, index=gene_names, columns=chunk_accessions)
            chunk_df = self._normalize_df(chunk_df, gene_lengths, canonical_genes, gene_name_map)

            if chunk_df.shape[1] == 0:
                print(f"{raw.shape[1]} bulk, 0 passed QC")
                continue

            # Write to disk instead of keeping in RAM
            batch_path = tmp / f"{species}_batch_{batch_num:04d}.parquet"
            chunk_df.to_parquet(batch_path, compression="zstd")
            batch_paths.append(batch_path)
            all_accessions.extend(chunk_df.columns.tolist())
            total += chunk_df.shape[1]

            elapsed = time.time() - t_species
            frac = b_end / n_total
            eta = (elapsed / frac - elapsed) if frac > 0 else 0
            print(f"+{chunk_df.shape[1]:,} QC pass ({total:,} kept / "
                  f"{total_bulk:,} bulk) "
                  f"[{elapsed:.0f}s, ~{eta:.0f}s left]")

            del raw, chunk_df
            gc.collect()

        h.close()

        metadata = pd.DataFrame({
            "geo_accession": all_accessions,
            "species": species,
        })
        print(f"    Species done in {time.time() - t_species:.0f}s — "
              f"{total:,} samples in {len(batch_paths)} batch files")
        return batch_paths, metadata


# ============================================================
# DATASET BUILDER (main API)
# ============================================================
class RNADatasetBuilder:
    """
    End-to-end RNA-seq preprocessing pipeline.
    Outputs a single concatenated dataset; train/val/test splitting
    is deferred to scikit-learn.

    Usage:
        builder = RNADatasetBuilder(species="both", gene_set="shared_orthologs")
        builder.process()
        builder.save_parquet("output/")
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None, **kwargs):
        """
        Args:
            config: Full PreprocessingConfig, OR pass individual kwargs:
                species, gene_set, train_samples, val_samples, etc.
        """
        if config is not None:
            self.config = config
        else:
            self.config = PreprocessingConfig(**kwargs)

        self.registry = GeneRegistry(
            self.config.orthologs_file,
            self.config.protein_coding_file,
        )
        self.loader = ExpressionLoader(self.config)

        # Populated by process()
        self.canonical_genes: list[str] = []
        self.batch_paths: list[Path] = []           # temp parquet files
        self.meta: Optional[pd.DataFrame] = None
        self.zero_genes: dict[str, set] = {}        # species → set of zero-expression genes

    def _species_list(self) -> list[str]:
        if self.config.species == "both":
            return ["human", "mouse"]
        return [self.config.species]

    def _h5_path(self, species: str) -> str:
        fname = "human_gene_v2.5.h5" if species == "human" else "mouse_gene_v2.5.h5"
        return os.path.join(self.config.archs4_dir, fname)

    def process(self):
        """Run the full pipeline: extract → normalize → find zero genes."""
        import shutil
        t0 = time.time()
        cfg = self.config
        species_list = self._species_list()

        # --- Settings ---
        print("=" * 70)
        print(f"PREPROCESSING SETTINGS")
        print(f"{'='*70}")
        print(f"  Species:           {cfg.species}")
        print(f"  Gene set:          {cfg.gene_set}")
        print(f"  Normalization:     {cfg.normalization}")
        print(f"  QC min nonzero:    {cfg.qc_min_nonzero:,}")
        print(f"  Remove SC:         {cfg.remove_single_cell}")
        print(f"  Batch size:        {cfg.extraction_batch_size:,}")
        print(f"  Output dir:        {cfg.output_dir}")
        if cfg.max_samples_per_species:
            print(f"  Max samples/sp:    {cfg.max_samples_per_species:,}")
        else:
            print(f"  Max samples/sp:    all")
        print(f"  Seed:              {cfg.seed}")
        print("=" * 70)

        # Start with all protein-coding orthologs
        all_ortho = self.registry.all_human_ortho
        print(f"\nProtein-coding orthologs: {len(all_ortho):,}")

        # --- Extract all samples using full gene list (stream to disk) ---
        tmp_dir = os.path.join(cfg.output_dir, "_tmp_batches")
        print(f"\n{'='*70}")
        print(f"Extracting all bulk samples (streaming to {tmp_dir})")
        print(f"{'='*70}")

        all_batch_paths = []
        all_metas = []
        species_batch_paths: dict[str, list[Path]] = {}

        for species in species_list:
            print(f"\n  [{species.upper()}]")
            # Mouse H5 uses mouse gene symbols; remap to human via orthologs
            name_map = self.registry.mouse_to_human if species == "mouse" else None
            batch_paths, meta = self.loader.extract_and_normalize(
                self._h5_path(species),
                species,
                all_ortho,
                tmp_dir,
                gene_name_map=name_map,
            )
            all_batch_paths.extend(batch_paths)
            species_batch_paths[species] = batch_paths
            if not meta.empty:
                all_metas.append(meta)

        if not all_batch_paths:
            print("No samples extracted.")
            return

        self.batch_paths = all_batch_paths
        self.meta = pd.concat(all_metas, ignore_index=True)
        total_samples = len(self.meta)
        print(f"\nTotal after extraction: {total_samples:,} samples × "
              f"{len(all_ortho):,} genes in {len(all_batch_paths)} batch files")

        # --- Remove zero-expression genes from actual data ---
        if cfg.gene_set == "shared_orthologs":
            print(f"\nFinding zero-expression genes (scanning batch files)...")
            for species in species_list:
                # Accumulate per-gene sums across all batches for this species
                gene_sum = None
                for bp in species_batch_paths[species]:
                    batch_df = pd.read_parquet(bp)
                    batch_sum = batch_df.sum(axis=1)
                    if gene_sum is None:
                        gene_sum = batch_sum
                    else:
                        gene_sum = gene_sum.add(batch_sum, fill_value=0)
                    del batch_df
                    gc.collect()
                if gene_sum is not None:
                    zero = set(gene_sum.index[gene_sum == 0])
                    self.zero_genes[species] = zero
                    print(f"  {species}: {len(zero):,} genes with zero expression")

            all_zero = set()
            for zg in self.zero_genes.values():
                all_zero |= zg
            self.canonical_genes = [g for g in all_ortho if g not in all_zero]
            print(f"  Removed {len(all_zero):,} genes → {len(self.canonical_genes):,} remaining")
        else:
            self.canonical_genes = list(all_ortho)

        print(f"\nFinal: {total_samples:,} samples × "
              f"{len(self.canonical_genes):,} genes")

        elapsed = time.time() - t0

        # --- Summary ---
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"  Species:         {cfg.species}")
        print(f"  Gene set:        {cfg.gene_set}")
        print(f"  Protein-coding:  yes ({len(self.registry.protein_coding):,} whitelist)")
        print(f"  Canonical genes: {len(self.canonical_genes):,}")
        print(f"  Total samples:   {total_samples:,}")
        for sp in species_list:
            sp_count = (self.meta["species"] == sp).sum() if self.meta is not None else 0
            print(f"    {sp:>8s}:      {sp_count:,}")
        print(f"  Normalization:   {cfg.normalization}")
        if self.zero_genes:
            print(f"  Zero-expression genes removed:")
            for sp, zg in self.zero_genes.items():
                print(f"    {sp:>8s}:      {len(zg):,}")
        print(f"  Batch files:     {len(self.batch_paths)} in {tmp_dir}")
        print(f"  Time elapsed:    {elapsed / 60:.1f} min")
        print(f"{'='*70}")

    def save_parquet(self, output_dir: Optional[str] = None):
        """
        Merge batch parquets into final output, filtering to canonical genes.

        Output:
            {output_dir}/expression.parquet   (genes × samples, float32, zstd)
            {output_dir}/metadata.csv          (geo_accession, species)
            {output_dir}/canonical_genes.csv   (token_id, gene_symbol)
        """
        import shutil
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Canonical gene list
        gene_df = pd.DataFrame({
            "token_id": range(1, len(self.canonical_genes) + 1),
            "gene_symbol": self.canonical_genes,
        })
        gene_df.to_csv(out / "canonical_genes.csv", index=False)

        with open(out / "genes.json", "w") as f:
            json.dump(self.canonical_genes, f)

        samples = self.meta["geo_accession"].tolist()
        with open(out / "samples.json", "w") as f:
            json.dump(samples, f)

        print(f"Saved canonical_genes.csv, genes.json ({len(self.canonical_genes):,} genes), "
              f"samples.json ({len(samples):,} samples)")

        # Merge batch parquets → one final parquet (streaming, low memory)
        print(f"Merging {len(self.batch_paths)} batch files into expression.parquet...",
              flush=True)
        merged_parts = []
        for i, bp in enumerate(self.batch_paths):
            batch_df = pd.read_parquet(bp)
            # Filter to canonical genes (drop zero-expression genes)
            batch_df = batch_df.loc[self.canonical_genes]
            merged_parts.append(batch_df)

            # Merge in chunks of 10 to limit memory
            if len(merged_parts) >= 10:
                merged_parts = [pd.concat(merged_parts, axis=1)]
                gc.collect()

            if (i + 1) % 20 == 0 or (i + 1) == len(self.batch_paths):
                print(f"  {i + 1}/{len(self.batch_paths)} batch files read", flush=True)

        combined = pd.concat(merged_parts, axis=1)
        expr_path = out / "expression.parquet"
        combined.to_parquet(expr_path, compression="zstd")
        print(f"Saved {expr_path}: {combined.shape}")
        del combined, merged_parts
        gc.collect()

        # Metadata
        meta_path = out / "metadata.csv"
        self.meta.to_csv(meta_path, index=False)
        print(f"Saved {meta_path}: {len(self.meta):,} rows")

        # Clean up temp batch files
        tmp_dir = Path(self.config.output_dir) / "_tmp_batches"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            print(f"Cleaned up {tmp_dir}")

        print("Done.")

    def get_data(self) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Get processed data as a numpy array by reading batch parquets.

        Returns:
            (X, metadata) where X is [samples, genes] float32 array
        """
        parts = []
        for bp in self.batch_paths:
            batch_df = pd.read_parquet(bp)
            batch_df = batch_df.loc[self.canonical_genes]
            parts.append(batch_df)
        combined = pd.concat(parts, axis=1)
        X = combined.values.T.astype(np.float32)  # [samples, genes]
        return X, self.meta


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RNA-seq preprocessing pipeline")
    parser.add_argument("--species", default="both", choices=["human", "mouse", "both"])
    parser.add_argument("--gene-set", default="shared_orthologs",
                        choices=["shared_orthologs", "union_orthologs"])
    parser.add_argument("--output-dir", default="data/archs4/train_orthologs")
    parser.add_argument("--qc-min-nonzero", type=int, default=14_000)
    parser.add_argument("--normalization", default="tpm",
                        choices=["log1p_tpm", "tpm", "raw_counts"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per species (default: all). "
                             "Useful for quick test runs.")
    args = parser.parse_args()

    config = PreprocessingConfig(
        species=args.species,
        gene_set=args.gene_set,
        output_dir=args.output_dir,
        qc_min_nonzero=args.qc_min_nonzero,
        normalization=args.normalization,
        max_samples_per_species=args.max_samples,
    )

    builder = RNADatasetBuilder(config=config)
    builder.process()
    builder.save_parquet()
