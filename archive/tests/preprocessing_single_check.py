import argparse
import random
import json
import bisect
from collections import Counter
from pathlib import Path

import archs4py as a4
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def section(title: str) -> None:
	print(f"\n{title}")
	print("-" * len(title))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Validate that a single expression.parquet matches H5-derived TPM values after "
			"the same ortholog and filtering logic used by preprocessing.py."
		)
	)
	parser.add_argument("--n-human", type=int, default=3, help="Number of human samples to test")
	parser.add_argument("--n-mouse", type=int, default=3, help="Number of mouse samples to test")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
	parser.add_argument("--threshold", type=float, default=0.98, help="Minimum Pearson r to pass")
	parser.add_argument(
		"--top-k-fail",
		type=int,
		default=20,
		help="Number of top abs-diff genes to print on failure",
	)
	parser.add_argument(
		"--base-dir",
		type=str,
		default="..",
		help="Workspace base dir when script is run from tests/",
	)
	parser.add_argument(
		"--parquet-relpath",
		type=str,
		default="data/archs4/train_orthologs_merged/expression.parquet",
		help="Parquet path relative to base-dir",
	)
	return parser.parse_args()


def preprocess_h5(
	df_raw: pd.DataFrame,
	canonical_genes: list[str],
	gene_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
	# Sum duplicate gene rows (isoforms/duplicates)
	df_proc = df_raw.T.groupby(level=0).sum().T

	# Remap gene names (e.g., mouse -> human), then sum collisions
	if gene_name_map is not None:
		mapped_cols = [gene_name_map.get(g, g) for g in df_proc.columns]
		df_proc.columns = mapped_cols
		df_proc = df_proc.T.groupby(level=0).sum().T

	# Final align to canonical ortholog gene list
	df_proc = df_proc.reindex(columns=canonical_genes, fill_value=0.0)
	return df_proc


def tpm_normalize(df: pd.DataFrame, exon_lengths: pd.Series) -> pd.DataFrame:
	# Mirror preprocessing.py: drop genes without exon lengths before TPM.
	lengths_bp = exon_lengths.reindex(df.columns)
	missing_mask = lengths_bp.isna()
	n_missing = int(missing_mask.sum())
	if n_missing > 0:
		print(
			f"[TPM] Missing exon lengths for {n_missing:,} genes in test matrix; "
			f"filtering them out before TPM"
		)

	keep_genes = lengths_bp.index[~missing_mask]
	df_use = df.loc[:, keep_genes]
	lengths_kb = lengths_bp.loc[keep_genes] / 1000.0

	if df_use.shape[1] == 0:
		return df.reindex(columns=df.columns, fill_value=0.0).astype(float)

	rate = df_use.astype(float).div(lengths_kb, axis=1)
	denom = rate.sum(axis=1).replace(0, np.nan)

	print(
		f"[TPM DEBUG] genes_used={rate.shape[1]:,}, samples={rate.shape[0]:,}, "
		f"denom(min/mean/max)={denom.min():.6f}/{denom.mean():.6f}/{denom.max():.6f}"
	)
	for sid, d in denom.head(3).items():
		print(f"[TPM DEBUG] sample={sid} denom={d:.6f}")

	tpm = rate.div(denom, axis=0) * 1e6
	return tpm.reindex(columns=df.columns, fill_value=0.0).fillna(0.0)


def compare_tpm(
	df_parquet: pd.DataFrame,
	df_h5_tpm: pd.DataFrame,
	sample_ids: list[str],
	label: str,
	threshold: float,
	top_k_fail: int,
) -> list[str]:
	failures: list[str] = []

	for i, sid in enumerate(sample_ids, start=1):
		print(f"\n[{label}] Sample {i}/{len(sample_ids)}: {sid}")
		common_genes = [g for g in df_parquet.columns if g in df_h5_tpm.columns]
		x = df_parquet.loc[sid, common_genes]
		y = df_h5_tpm.loc[sid, common_genes]
		corr_all = float(np.corrcoef(x, y)[0, 1])

		sum_x = float(x.sum())
		sum_y = float(y.sum())
		ratio_sum = sum_x / sum_y if sum_y != 0 else np.nan
		print(f"{label} sample {sid}: sum(parquet)={sum_x:.4f}, sum(h5_tpm)={sum_y:.4f}, ratio={ratio_sum:.6f}")

		both_nonzero = (x > 0) & (y > 0)
		x_only_nonzero = (x > 0) & (y == 0)
		y_only_nonzero = (x == 0) & (y > 0)

		corr_both_nonzero = np.nan
		if int(both_nonzero.sum()) >= 2:
			corr_both_nonzero = float(np.corrcoef(x[both_nonzero], y[both_nonzero])[0, 1])

		if int(both_nonzero.sum()) > 0:
			scale = (x[both_nonzero] / y[both_nonzero]).replace([np.inf, -np.inf], np.nan).dropna()
			if len(scale) > 0:
				median_ratio = float(np.median(scale))
				p10 = float(np.percentile(scale, 10))
				p90 = float(np.percentile(scale, 90))
				print(
					f"{label} sample {sid}: scale ratio parquet/h5 median={median_ratio:.4f} "
					f"(p10={p10:.4f}, p90={p90:.4f})"
				)

		print(
			f"{label} sample {sid}: both_nonzero={int(both_nonzero.sum())}, "
			f"parquet_only_nonzero={int(x_only_nonzero.sum())}, "
			f"h5_only_nonzero={int(y_only_nonzero.sum())}"
		)

		print(f"{label} sample {sid}: Pearson r (all genes) = {corr_all:.4f}")
		if np.isnan(corr_both_nonzero):
			print(f"{label} sample {sid}: Pearson r (both-nonzero genes) = NA (insufficient overlap)")
		else:
			print(f"{label} sample {sid}: Pearson r (both-nonzero genes) = {corr_both_nonzero:.4f}")

		if corr_all >= threshold:
			print("PASS: Correlation above threshold (all genes).")
		else:
			print("FAIL: Correlation below threshold (all genes)!")
			failures.append(f"{label}:{sid}:r={corr_all:.6f}")

			diff = (x - y).abs().sort_values(ascending=False).head(top_k_fail)
			diff_df = pd.DataFrame(
				{
					"Gene": diff.index,
					"Parquet": x.loc[diff.index].values,
					"H5_TPM": y.loc[diff.index].values,
					"AbsDiff": diff.values,
				}
			)
			print(f"{label} sample {sid}: Top {top_k_fail} absolute-difference genes")
			print(diff_df.to_string(index=False))

		print()

	return failures


def _get_parquet_layout(expression_parquet: Path):
	pf = pq.ParquetFile(str(expression_parquet))
	cols = pf.schema_arrow.names
	idx_col = "geo_accession" if "geo_accession" in cols else (
		"__index_level_0__" if "__index_level_0__" in cols else None
	)
	if idx_col is None:
		raise ValueError(
			"Parquet is missing sample index column. Expected geo_accession or __index_level_0__."
		)

	gene_cols = [c for c in cols if c not in ("geo_accession", "__index_level_0__")]
	id_table = pf.read(columns=[idx_col], use_threads=True)
	all_ids = [str(x) for x in id_table.column(0).to_pylist()]
	id_to_row = {sid: i for i, sid in enumerate(all_ids)}

	starts = [0]
	for rg in range(pf.metadata.num_row_groups):
		starts.append(starts[-1] + pf.metadata.row_group(rg).num_rows)

	return pf, idx_col, gene_cols, all_ids, id_to_row, starts


def _load_selected_rows(
	pf: pq.ParquetFile,
	idx_col: str,
	gene_cols: list[str],
	id_to_row: dict[str, int],
	row_group_starts: list[int],
	sample_ids: list[str],
) -> pd.DataFrame:
	grouped = {}
	for sid in sample_ids:
		global_row = id_to_row[sid]
		rg_idx = bisect.bisect_right(row_group_starts, global_row) - 1
		local_row = global_row - row_group_starts[rg_idx]
		grouped.setdefault(rg_idx, []).append((sid, local_row))

	parts = []
	for rg_idx, reqs in grouped.items():
		local_rows = [r for _, r in reqs]
		rg_table = pf.read_row_group(rg_idx, columns=[idx_col] + gene_cols, use_threads=True)
		sub = rg_table.take(pa.array(local_rows, type=pa.int64()))
		sub_df = sub.to_pandas()

		# Parquet written from pandas may restore sample IDs as index instead of a column.
		if idx_col in sub_df.columns:
			sub_df[idx_col] = sub_df[idx_col].astype(str)
			sub_df = sub_df.set_index(idx_col)
		elif sub_df.index.name in (idx_col, "geo_accession", "__index_level_0__"):
			sub_df.index = sub_df.index.astype(str)
			sub_df.index.name = idx_col
		elif "geo_accession" in sub_df.columns:
			sub_df["geo_accession"] = sub_df["geo_accession"].astype(str)
			sub_df = sub_df.set_index("geo_accession")
		elif "__index_level_0__" in sub_df.columns:
			sub_df["__index_level_0__"] = sub_df["__index_level_0__"].astype(str)
			sub_df = sub_df.set_index("__index_level_0__")
		else:
			raise KeyError(
				f"Could not find sample ID field. idx_col={idx_col}, "
				f"columns={list(sub_df.columns)}, index_name={sub_df.index.name}"
			)
		parts.append(sub_df)

	df = pd.concat(parts, axis=0)
	df = df.loc[sample_ids]
	return df


def main() -> int:
	args = parse_args()

	base_dir = Path(args.base_dir)
	expression_parquet = base_dir / args.parquet_relpath
	samples_meta_path = base_dir / "data/archs4/train_orthologs/samples.json"
	human_h5 = base_dir / "data/archs4/human_gene_v2.5.h5"
	mouse_h5 = base_dir / "data/archs4/mouse_gene_v2.5.h5"
	protein_coding_genes_path = base_dir / "data/ensembl/protein_coding_ortholog_genes.txt"
	orthologs_path = base_dir / "data/ensembl/orthologs_one2one.txt"
	human_len_path = base_dir / "data/gencode/gencode_v49_gene_exon_lengths.csv"
	mouse_len_path = base_dir / "data/gencode/gencode_v49_mouse_gene_exon_lengths.csv"

	section("[LOAD] Reading samples metadata and single parquet")

	with open(samples_meta_path) as f:
		samples_list = json.load(f)
	meta = pd.DataFrame(samples_list)
	meta = meta.rename(columns={"id": "geo_accession"})

	if not expression_parquet.exists():
		raise FileNotFoundError(f"No parquet file found at {expression_parquet}")

	pf, idx_col, gene_cols, all_ids, id_to_row, row_group_starts = _get_parquet_layout(expression_parquet)
	loaded_samples = set(all_ids)
	print(f"Loaded single parquet: {expression_parquet}")
	print(f"Discovered shape: {len(all_ids)} samples x {len(gene_cols)} genes")

	section("[ID DIAGNOSTICS] Checking sample-ID uniqueness")
	meta_dup_count = int(meta["geo_accession"].duplicated().sum())
	parquet_dup_count = len(all_ids) - len(loaded_samples)
	print(f"Metadata duplicate geo_accession rows: {meta_dup_count}")
	print(f"Parquet duplicate sample IDs: {parquet_dup_count}")

	if meta_dup_count > 0:
		dup_ids = meta.loc[meta["geo_accession"].duplicated(), "geo_accession"].head(10).tolist()
		raise ValueError(f"Duplicate metadata geo_accession IDs detected (first 10): {dup_ids}")
	if parquet_dup_count > 0:
		counts = Counter(all_ids)
		dup_ids = [sid for sid, c in counts.items() if c > 1][:10]
		raise ValueError(f"Duplicate parquet sample IDs detected (first 10): {dup_ids}")

	human_pool = meta[meta["species"] == "human"]["geo_accession"].tolist()
	mouse_pool = meta[meta["species"] == "mouse"]["geo_accession"].tolist()

	human_in_loaded = [h for h in human_pool if h in loaded_samples]
	mouse_in_loaded = [m for m in mouse_pool if m in loaded_samples]

	random.seed(args.seed)
	n_h = min(args.n_human, len(human_in_loaded))
	n_m = min(args.n_mouse, len(mouse_in_loaded))
	human_sample_ids = random.sample(human_in_loaded, n_h) if n_h > 0 else []
	mouse_sample_ids = random.sample(mouse_in_loaded, n_m) if n_m > 0 else []
	test_sample_ids = human_sample_ids + mouse_sample_ids

	section("[SAMPLE SELECTION]")
	print(f"Selected {len(test_sample_ids)} samples from single parquet:")
	for sid in human_sample_ids:
		print(f"  {sid:20s} (human)")
	for sid in mouse_sample_ids:
		print(f"  {sid:20s} (mouse)")

	if not test_sample_ids:
		raise RuntimeError("No eligible test samples found in single parquet")

	# Load only selected rows from parquet to keep memory usage low.
	selected_ids = human_sample_ids + mouse_sample_ids
	df_selected = _load_selected_rows(
		pf,
		idx_col,
		gene_cols,
		id_to_row,
		row_group_starts,
		selected_ids,
	)
	df_human = df_selected.loc[human_sample_ids] if human_sample_ids else pd.DataFrame(columns=gene_cols)
	df_mouse = df_selected.loc[mouse_sample_ids] if mouse_sample_ids else pd.DataFrame(columns=gene_cols)

	section("[CHECK] Single parquet extraction")
	print("PASS: All human samples found in parquet." if set(human_sample_ids).issubset(loaded_samples) else "FAIL: Some human samples missing in parquet.")
	print("PASS: All mouse samples found in parquet." if set(mouse_sample_ids).issubset(loaded_samples) else "FAIL: Some mouse samples missing in parquet.")

	df_human_raw = a4.data.samples(str(human_h5), human_sample_ids).T if human_sample_ids else pd.DataFrame()
	found_human = df_human_raw.index.intersection(human_sample_ids).tolist() if human_sample_ids else []
	missing_human = list(set(human_sample_ids) - set(found_human))

	section("[H5 LOOKUP] Selected sample availability")
	print(f"[HUMAN] Found in H5: {found_human}")
	print(f"[HUMAN] Missing in H5: {missing_human}")

	df_mouse_raw = a4.data.samples(str(mouse_h5), mouse_sample_ids).T if mouse_sample_ids else pd.DataFrame()
	found_mouse = df_mouse_raw.index.intersection(mouse_sample_ids).tolist() if mouse_sample_ids else []
	missing_mouse = list(set(mouse_sample_ids) - set(found_mouse))

	print(f"\n[MOUSE] Found in H5: {found_mouse}")
	print(f"[MOUSE] Missing in H5: {missing_mouse}")

	if missing_human or missing_mouse:
		raise RuntimeError("One or more selected sample IDs were not found in H5")

	section("[PREPROCESSING] Loading ortholog maps and exon lengths")
	with open(protein_coding_genes_path) as f:
		human_pc_genes = set(line.strip() for line in f if line.strip())

	orthologs = pd.read_csv(orthologs_path, sep="\t", header=0)
	orthologs = orthologs[orthologs["Human homology type"] == "ortholog_one2one"]
	orthologs = orthologs[orthologs["Human gene name"].isin(human_pc_genes)]

	human_ortholog_genes = set(orthologs["Human gene name"])
	mouse_to_human = dict(zip(orthologs["Gene name"], orthologs["Human gene name"]))

	human_lengths = pd.read_csv(human_len_path).set_index("gene_symbol")["exon_length"]
	mouse_lengths = pd.read_csv(mouse_len_path).set_index("gene_symbol")["exon_length"]
	mouse_lengths_human_space = mouse_lengths.rename(index=mouse_to_human).groupby(level=0).first()

	canonical_genes = [g for g in gene_cols if g in human_ortholog_genes]
	human_tpm_genes = [g for g in canonical_genes if g in human_lengths.index]
	mouse_tpm_genes = [g for g in canonical_genes if g in mouse_lengths_human_space.index]

	print(f"[TPM GENE SET] Human genes with exon lengths: {len(human_tpm_genes):,}")
	print(f"[TPM GENE SET] Mouse genes with exon lengths (human space): {len(mouse_tpm_genes):,}")

	df_human_h5_proc = preprocess_h5(df_human_raw, human_tpm_genes)
	df_mouse_h5_proc = preprocess_h5(df_mouse_raw, mouse_tpm_genes, gene_name_map=mouse_to_human)

	df_human = df_human[human_tpm_genes]
	df_mouse = df_mouse[mouse_tpm_genes]

	section("[TPM NORMALIZATION] Normalizing H5-processed data")
	df_human_h5_tpm = tpm_normalize(df_human_h5_proc, human_lengths)
	df_mouse_h5_tpm = tpm_normalize(df_mouse_h5_proc, mouse_lengths_human_space)

	section("[METRIC COMPARISON] Pearson correlation between H5 TPM and parquet values")
	failures = []
	failures.extend(compare_tpm(df_human, df_human_h5_tpm, found_human, "Human", args.threshold, args.top_k_fail))
	failures.extend(compare_tpm(df_mouse, df_mouse_h5_tpm, found_mouse, "Mouse", args.threshold, args.top_k_fail))

	print(f"\n[SUMMARY] Genes in parquet: {len(gene_cols):,}")
	print(f"[SUMMARY] Genes used for TPM comparison (human): {len(human_tpm_genes):,}")

	if failures:
		print("\n[RESULT] FAIL")
		for item in failures:
			print(f"  - {item}")
		return 1

	print("\n[RESULT] PASS")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
