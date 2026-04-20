import s3fs
import h5py
import numpy as np

HUMAN_S3 = "mssm-seq-matrix/human_matrix_v11.h5"
MOUSE_S3 = "mssm-seq-matrix/mouse_matrix_v11.h5"

def check_structure(s3_path):
    print(f"\nChecking: {s3_path}")
    s3 = s3fs.S3FileSystem(anon=True)
    with h5py.File(s3.open(s3_path, "rb"), "r") as f:
        # Check shape
        expr = f["data/expression"]
        print(f"  data/expression shape: {expr.shape}")

        # Check required metadata keys
        keys_to_check = [
            "meta/genes/symbol",
            "meta/samples/geo_accession",
            "meta/samples/singlecellprobability",
        ]
        for k in keys_to_check:
            if k in f:
                val = f[k]
                sample = f[k][0]
                if isinstance(sample, bytes):
                    sample = sample.decode()
                print(f"  {k}: len={len(val)}, sample={sample}")
            else:
                print(f"  MISSING: {k}")

        # Test reading 3 random samples
        print("  Reading 3 sample columns...", end=" ", flush=True)
        test = expr[:, :3]
        print(f"ok, shape={test.shape}, dtype={test.dtype}")

check_structure(HUMAN_S3)
check_structure(MOUSE_S3)
