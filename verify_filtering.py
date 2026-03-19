#!/usr/bin/env python3
"""Verify that metadata columns are filtered out correctly"""

import pyarrow.parquet as pq
from pathlib import Path

batch_dir = Path("data/archs4/train_orthologs/batch_files")
batch_file = sorted(batch_dir.glob("*.parquet"))[0]

parquet_file = pq.ParquetFile(str(batch_file))
all_cols = parquet_file.schema_arrow.names
sample_ids = [s for s in all_cols if s not in {'gene_symbol', 'gene_id'}]

print(f"File: {batch_file.name}")
print(f"Total columns in schema: {len(all_cols)}")
print(f"Columns that are actual samples: {len(sample_ids)}")
print(f"\nLast 5 columns (before filtering):")
for col in all_cols[-5:]:
    print(f"  - '{col}'")
print(f"\nMetadata columns filtered out: {set(all_cols) - set(sample_ids)}")
