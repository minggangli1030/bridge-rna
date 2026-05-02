"""
Compute the shared canonical gene vocabulary used by all 5k_v2 variants.

Vocab = protein_coding_ortholog_genes ∩ human_exon_lengths ∩ mouse_exon_lengths
        (mouse genes mapped to human symbols via the one-to-one ortholog table).

This is a *data-independent* filter — it does not depend on which samples were
drawn for any given variant. Passing the resulting file to all variants via
--canonical-genes-file guarantees identical column order, fixing the bug that
broke the original MoE PoC.

Run locally and commit data/ensembl/canonical_genes_shared.txt.
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
ORTHOLOGS = REPO / "data/ensembl/orthologs_one2one.txt"
PROTEIN_CODING = REPO / "data/ensembl/protein_coding_ortholog_genes.txt"
HUMAN_LENGTHS = REPO / "data/gencode/gencode_v49_gene_exon_lengths.csv"
MOUSE_LENGTHS = REPO / "data/gencode/gencode_v49_mouse_gene_exon_lengths.csv"
OUT = REPO / "data/ensembl/canonical_genes_shared.txt"


def main() -> None:
    with open(PROTEIN_CODING) as f:
        protein_coding = {line.strip() for line in f if line.strip()}

    ortho = pd.read_csv(ORTHOLOGS, sep="\t")
    mouse_to_human = dict(zip(ortho["Gene name"], ortho["Human gene name"]))
    all_human_ortho = sorted(
        g for g in ortho["Human gene name"].unique()
        if isinstance(g, str) and g in protein_coding
    )

    human_lens = pd.read_csv(HUMAN_LENGTHS).set_index("gene_symbol")["exon_length"]
    mouse_lens_raw = pd.read_csv(MOUSE_LENGTHS).set_index("gene_symbol")["exon_length"]
    # Map mouse symbols → human via the one-to-one ortholog table; collapse dups.
    mouse_lens_human = (
        mouse_lens_raw.rename(index=mouse_to_human)
        .groupby(level=0)
        .first()
    )

    valid_len_genes = set(human_lens.index) & set(mouse_lens_human.index)

    canonical = [g for g in all_human_ortho if g in valid_len_genes]

    print(f"protein_coding_orthologs : {len(all_human_ortho):,}")
    print(f"human exon lengths       : {len(human_lens):,}")
    print(f"mouse exon lengths (h-sym): {len(mouse_lens_human):,}")
    print(f"intersection (canonical) : {len(canonical):,}")

    OUT.write_text("\n".join(canonical) + "\n")
    print(f"\nWrote {OUT} ({len(canonical)} genes)")


if __name__ == "__main__":
    main()
