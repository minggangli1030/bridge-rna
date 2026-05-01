"""Diagnose the train_moe.py gene-count mismatch.

For each 5k expert: print the gene count baked into the saved gene_embedding
weights, the value (if any) recorded in the checkpoint config, and the column
count of the parquet train_moe.py would feed it. A mismatch between the first
and third is exactly what made job 33939250 fail at load_state_dict.

Run from repo root on Savio:
    cd /global/scratch/users/minggangli/bridge-rna
    python check_moe_gene_counts.py
"""
from pathlib import Path
import torch
import pyarrow.parquet as pq

VARIANTS = ["human_5k", "mouse_5k", "mixed_5k"]
NON_GENE_COLS = {"geo_accession", "__index_level_0__", "sample_id"}


def parquet_gene_cols(path: Path) -> list[str]:
    return [c for c in pq.ParquetFile(str(path)).schema_arrow.names if c not in NON_GENE_COLS]


def main() -> None:
    rows = []
    for name in VARIANTS:
        ckpt_path = Path(f"checkpoints/{name}/best_model.pt")
        pq_path = Path(f"data/archs4/{name}_merged/expression.parquet")

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        n_emb = sd["gene_embedding.weight"].shape[0]
        cfg_n = ck.get("config", {}).get("num_genes", "<missing>")

        if pq_path.exists():
            cols = parquet_gene_cols(pq_path)
            n_pq = len(cols)
        else:
            cols = None
            n_pq = "<missing>"

        rows.append((name, n_emb, cfg_n, n_pq, ckpt_path, pq_path, cols))
        print(f"{name}: ckpt_emb={n_emb}  cfg.num_genes={cfg_n}  parquet_cols={n_pq}")

    print()
    embs = {r[1] for r in rows}
    if len(embs) == 1:
        print(f"All experts agree on gene count: {next(iter(embs))}")
    else:
        print(f"WARNING: experts disagree on gene count: {embs}")
        print("  -> the PoC's shared-gene-space assumption is broken.")

    valid_pq = [r for r in rows if isinstance(r[3], int)]
    if valid_pq and len({r[3] for r in valid_pq}) == 1:
        n_pq_common = valid_pq[0][3]
        if len(embs) == 1 and next(iter(embs)) != n_pq_common:
            print(
                f"\nMismatch: experts trained on {next(iter(embs))} genes, "
                f"current parquets have {n_pq_common}."
            )
            print("  -> parquets were regenerated after experts were trained.")
            print("  -> need the original gene order to re-index, or retrain experts.")

    cols_sets = [r[6] for r in rows if r[6] is not None]
    if len(cols_sets) >= 2:
        ref = cols_sets[0]
        for other, r in zip(cols_sets[1:], rows[1:]):
            if other != ref:
                first_diff = next(
                    (i for i, (a, b) in enumerate(zip(ref, other)) if a != b), -1
                )
                print(
                    f"\nWARNING: parquet column order differs between "
                    f"{rows[0][0]} and {r[0]} at index {first_diff}"
                )
                break
        else:
            print("\nParquet column order is identical across the 3 variants.")

    print("\nLooking for stored gene-order artifacts under checkpoints/...")
    for name in VARIANTS:
        ckpt_dir = Path(f"checkpoints/{name}")
        if not ckpt_dir.exists():
            continue
        candidates = sorted(
            p for p in ckpt_dir.iterdir()
            if p.is_file() and p.name != "best_model.pt"
        )
        print(f"  {ckpt_dir}: {[p.name for p in candidates] or '<no extra files>'}")


if __name__ == "__main__":
    main()
