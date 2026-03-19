#!/usr/bin/env python3
"""Quick diagnostic to check parquet structure"""

import pyarrow.parquet as pq
from pathlib import Path

batch_dir = Path("data/archs4/train_orthologs/batch_files")
batch_files = sorted(batch_dir.glob("*.parquet"))

print(f"Found {len(batch_files)} batch files\n")

if batch_files:
    # Check first file
    first_file = batch_files[0]
    print(f"Checking: {first_file.name}")
    
    # Read with no columns
    table = pq.read_table(str(first_file), columns=[])
    print(f"\nWith columns=[]: {len(table.column_names)} columns")
    print(f"Column names (first 10): {table.column_names[:10]}")
    
    # Read schema
    parquet_file = pq.ParquetFile(str(first_file))
    schema = parquet_file.schema
    print(f"\nSchema: {schema}")
    print(f"Column names from schema (first 10): {schema.names[:10]}")
    print(f"Total columns in schema: {len(schema.names)}")
    
    # Check metadata
    metadata = parquet_file.metadata
    print(f"\nMetadata: {metadata}")
    
    # Try reading one column
    print("\n--- Attempting to read actual data ---")
    try:
        table2 = pq.read_table(str(first_file))
        print(f"Full read succeeded! Shape: {table2.shape}")
        print(f"Column names (first 10): {table2.column_names[:10]}")
    except Exception as e:
        print(f"Full read failed: {e}")
