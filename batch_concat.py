import pandas as pd
from pathlib import Path
import gc
import os

batch_dir = Path("data/archs4/train_orthologs/_tmp_batches")
output_file = Path("data/archs4/train_orthologs/expression.parquet")

batch_files = sorted(batch_dir.glob("*.parquet"))
print(f"Merging {len(batch_files)} batch files (streaming)...")

# Process and write each chunk immediately, don't hold in memory
chunk_size = 20
first_chunk = True

for start_idx in range(0, len(batch_files), chunk_size):
    end_idx = min(start_idx + chunk_size, len(batch_files))
    chunk_files = batch_files[start_idx:end_idx]
    
    parts = []
    for f in chunk_files:
        df = pd.read_parquet(f)
        parts.append(df)
    
    chunk_df = pd.concat(parts, axis=1)
    
    if first_chunk:
        chunk_df.to_parquet(str(output_file), compression="zstd")
        first_chunk = False
    else:
        # Append to existing file
        existing = pd.read_parquet(output_file)
        combined = pd.concat([existing, chunk_df], axis=1)
        combined.to_parquet(str(output_file), compression="zstd")
        del existing, combined
    
    del parts, chunk_df
    gc.collect()
    
    print(f"  Processed {end_idx}/{len(batch_files)} files")

print(f"Done: {output_file}")
print(f"Size: {os.path.getsize(output_file) / 1e9:.1f} GB")

# Cleanup
import shutil
shutil.rmtree(batch_dir)
print(f"Cleaned up {batch_dir}")

