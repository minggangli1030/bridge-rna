#!/usr/bin/env python3
"""Test the correct way to get column names from parquet"""

import pyarrow.parquet as pq
from pathlib import Path

batch_dir = Path("data/archs4/train_orthologs/batch_files")
batch_files = sorted(batch_dir.glob("*.parquet"))

if batch_files:
    first_file = batch_files[0]
    print(f"Testing: {first_file.name}\n")
    
    # WRONG approach
    table = pq.read_table(str(first_file), columns=[])
    print(f"❌ WRONG - read_table with columns=[]: {len(table.column_names)} columns")
    
    # CORRECT approach - use schema_arrow
    parquet_file = pq.ParquetFile(str(first_file))
    arrow_schema = parquet_file.schema_arrow
    sample_ids = arrow_schema.names
    
    print(f"✓ CORRECT - schema_arrow.names: {len(sample_ids)} columns")
    print(f"First 10 sample IDs: {sample_ids[:10]}")
    print(f"Last 5 sample IDs: {sample_ids[-5:]}")
