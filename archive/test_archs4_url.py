import s3fs
import h5py

HUMAN_S3 = "mssm-seq-matrix/human_matrix_v11.h5"

print("Checking meta/genes subkeys...")
s3 = s3fs.S3FileSystem(anon=True)
with h5py.File(s3.open(HUMAN_S3, "rb"), "r") as f:
    # List everything under meta/genes
    print("  meta/genes keys:", list(f["meta/genes"].keys()))

    # Show a sample from each key
    for k in f["meta/genes"].keys():
        val = f[f"meta/genes/{k}"]
        sample = val[0]
        if isinstance(sample, bytes):
            sample = sample.decode()
        print(f"  meta/genes/{k}: len={len(val)}, sample={repr(sample)}")
