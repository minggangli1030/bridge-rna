#!/usr/bin/env python3
"""
Generate all reference files needed by bridge-rna preprocessing.py.

Outputs (relative to script location):
  data/ensembl/orthologs_one2one.txt          mouse↔human one-to-one orthologs (TSV)
  data/ensembl/protein_coding_ortholog_genes.txt  (copied from sp26_nasa if path given)
  data/gencode/gencode_v49_gene_exon_lengths.csv  human exon lengths
  data/gencode/gencode_v49_mouse_gene_exon_lengths.csv  mouse exon lengths

Usage:
  # Generate all files (downloads from Ensembl BioMart):
  python fetch_reference_data.py

  # Also copy human exon lengths from sp26_nasa instead of downloading:
  python fetch_reference_data.py \
      --human-exon-lengths ../sp26_nasa/data_preprocessing/human_exon_lengths_df.csv \
      --protein-coding ../sp26_nasa/data_preprocessing/protein_coding_ortholog_genes.txt
"""

import argparse
import io
import shutil
import time
from pathlib import Path

import pandas as pd
import requests


BIOMART_URL = "https://www.ensembl.org/biomart/martservice"


def _biomart_query(xml: str, timeout: int = 300) -> pd.DataFrame:
    print("  Querying Ensembl BioMart...", flush=True)
    resp = requests.post(BIOMART_URL, data={"query": xml}, timeout=timeout)
    resp.raise_for_status()
    raw = resp.text.strip()
    if raw.startswith("ERROR") or raw.startswith("<"):
        raise RuntimeError(f"BioMart error response: {raw[:300]}")
    df = pd.read_csv(io.StringIO(raw), sep="\t", header=0)
    print(f"  Got {len(df):,} rows, columns: {df.columns.tolist()}")
    return df


def fetch_orthologs(output_path: Path) -> None:
    """Download mouse↔human one-to-one orthologs from Ensembl BioMart."""
    print("\n[1/4] Fetching mouse↔human one-to-one orthologs...")
    # Query mouse dataset for human homologs + orthology type
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="1"'
        '       uniqueRows="1" count="" datasetConfigVersion="0.6">'
        '<Dataset name="mmusculus_gene_ensembl" interface="default">'
        '<Attribute name="external_gene_name"/>'
        '<Attribute name="hsapiens_homolog_associated_gene_name"/>'
        '<Attribute name="hsapiens_homolog_orthology_type"/>'
        '</Dataset>'
        '</Query>'
    )

    df = _biomart_query(xml)

    # Normalize column names regardless of what BioMart returns
    df.columns = [c.strip() for c in df.columns]
    mouse_col = df.columns[0]
    human_col = df.columns[1]
    type_col  = df.columns[2] if len(df.columns) > 2 else None

    if type_col is not None:
        df = df[df[type_col] == "ortholog_one2one"].copy()

    df = df[[mouse_col, human_col]].copy()
    df.columns = ["Gene name", "Human gene name"]
    df = df.dropna()
    df = df[(df["Gene name"].str.strip() != "") & (df["Human gene name"].str.strip() != "")]
    df = df.drop_duplicates()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"  Saved {len(df):,} one-to-one ortholog pairs → {output_path}")


def fetch_mouse_exon_lengths(output_path: Path) -> None:
    """Download mouse gene exon lengths from Ensembl BioMart."""
    print("\n[2/4] Fetching mouse exon lengths from Ensembl BioMart...")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="1"'
        '       uniqueRows="0" count="" datasetConfigVersion="0.6">'
        '<Dataset name="mmusculus_gene_ensembl" interface="default">'
        '<Attribute name="external_gene_name"/>'
        '<Attribute name="exon_chrom_start"/>'
        '<Attribute name="exon_chrom_end"/>'
        '</Dataset>'
        '</Query>'
    )

    df = _biomart_query(xml)
    df.columns = ["gene_symbol", "exon_start", "exon_end"]
    df = df.dropna(subset=["gene_symbol", "exon_start", "exon_end"])
    df["exon_start"] = df["exon_start"].astype(int)
    df["exon_end"] = df["exon_end"].astype(int)
    df["exon_len"] = (df["exon_end"] - df["exon_start"]).abs() + 1

    # Sum exon lengths per gene (union of all exons, deduplicated by interval)
    # Simple approach: sum unique (start, end) pairs per gene
    df = df.drop_duplicates(subset=["gene_symbol", "exon_start", "exon_end"])
    lengths = df.groupby("gene_symbol")["exon_len"].sum().reset_index()
    lengths.columns = ["gene_symbol", "exon_length"]
    lengths = lengths[lengths["exon_length"] > 0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lengths.to_csv(output_path, index=False)
    print(f"  Saved {len(lengths):,} mouse gene exon lengths → {output_path}")


def convert_human_exon_lengths(input_path: Path, output_path: Path) -> None:
    """Convert sp26_nasa human_exon_lengths_df.csv to bridge-rna format."""
    print(f"\n[3/4] Converting human exon lengths from {input_path}...")
    df = pd.read_csv(input_path, index_col=0)
    # sp26_nasa uses 'gene name' and 'length'; bridge-rna expects 'gene_symbol' and 'exon_length'
    df = df.rename(columns={"gene name": "gene_symbol", "length": "exon_length"})
    df = df[["gene_symbol", "exon_length"]].dropna()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df):,} human gene exon lengths → {output_path}")


def fetch_human_exon_lengths(output_path: Path) -> None:
    """Download human gene exon lengths from Ensembl BioMart (fallback)."""
    print("\n[3/4] Fetching human exon lengths from Ensembl BioMart...")
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1"
       uniqueRows="0" count="" datasetConfigVersion="0.6">
    <Dataset name="hsapiens_gene_ensembl" interface="default">
        <Attribute name="external_gene_name"/>
        <Attribute name="exon_chrom_start"/>
        <Attribute name="exon_chrom_end"/>
    </Dataset>
</Query>"""

    df = _biomart_query(xml)
    df.columns = ["gene_symbol", "exon_start", "exon_end"]
    df = df.dropna(subset=["gene_symbol", "exon_start", "exon_end"])
    df["exon_start"] = df["exon_start"].astype(int)
    df["exon_end"] = df["exon_end"].astype(int)
    df["exon_len"] = (df["exon_end"] - df["exon_start"]).abs() + 1
    df = df.drop_duplicates(subset=["gene_symbol", "exon_start", "exon_end"])
    lengths = df.groupby("gene_symbol")["exon_len"].sum().reset_index()
    lengths.columns = ["gene_symbol", "exon_length"]
    lengths = lengths[lengths["exon_length"] > 0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lengths.to_csv(output_path, index=False)
    print(f"  Saved {len(lengths):,} human gene exon lengths → {output_path}")


def copy_protein_coding(input_path: Path, output_path: Path) -> None:
    print(f"\n[4/4] Copying protein coding gene list from {input_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    n = sum(1 for line in open(output_path) if line.strip())
    print(f"  Copied {n:,} genes → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch bridge-rna reference data files")
    parser.add_argument(
        "--human-exon-lengths",
        type=Path,
        default=None,
        help="Path to sp26_nasa human_exon_lengths_df.csv (skips BioMart download if given)",
    )
    parser.add_argument(
        "--protein-coding",
        type=Path,
        default=None,
        help="Path to protein_coding_ortholog_genes.txt to copy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Base output directory (default: ./data)",
    )
    args = parser.parse_args()

    out = args.output_dir
    orthologs_out   = out / "ensembl" / "orthologs_one2one.txt"
    prot_coding_out = out / "ensembl" / "protein_coding_ortholog_genes.txt"
    human_exon_out  = out / "gencode" / "gencode_v49_gene_exon_lengths.csv"
    mouse_exon_out  = out / "gencode" / "gencode_v49_mouse_gene_exon_lengths.csv"

    print("=" * 60)
    print("bridge-rna reference data fetcher")
    print("=" * 60)

    # 1. Orthologs (always from BioMart)
    if not orthologs_out.exists():
        fetch_orthologs(orthologs_out)
    else:
        print(f"\n[1/4] Skipping orthologs (already exists): {orthologs_out}")

    # 2. Mouse exon lengths (always from BioMart)
    if not mouse_exon_out.exists():
        fetch_mouse_exon_lengths(mouse_exon_out)
    else:
        print(f"\n[2/4] Skipping mouse exon lengths (already exists): {mouse_exon_out}")

    # 3. Human exon lengths
    if not human_exon_out.exists():
        if args.human_exon_lengths and args.human_exon_lengths.exists():
            convert_human_exon_lengths(args.human_exon_lengths, human_exon_out)
        else:
            fetch_human_exon_lengths(human_exon_out)
    else:
        print(f"\n[3/4] Skipping human exon lengths (already exists): {human_exon_out}")

    # 4. Protein coding gene list
    if not prot_coding_out.exists():
        if args.protein_coding and args.protein_coding.exists():
            copy_protein_coding(args.protein_coding, prot_coding_out)
        else:
            print("\n[4/4] WARNING: --protein-coding not provided and no file exists.")
            print("  Please provide path to protein_coding_ortholog_genes.txt")
    else:
        print(f"\n[4/4] Skipping protein coding list (already exists): {prot_coding_out}")

    print("\n" + "=" * 60)
    print("Done. Files generated:")
    for p in [orthologs_out, prot_coding_out, human_exon_out, mouse_exon_out]:
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
