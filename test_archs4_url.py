import archs4py as a4
import s3fs
import h5py

HUMAN_URL = "https://s3.amazonaws.com/mssm-seq-matrix/human_matrix_v11.h5"
MOUSE_URL = "https://s3.amazonaws.com/mssm-seq-matrix/mouse_matrix_v11.h5"

# Test rand_remote
print("Testing rand_remote with human_matrix_v11.h5...")
try:
    df = a4.data.rand_remote(HUMAN_URL, 3, remove_sc=True)
    print("  WORKS! Shape:", df.shape)
    print("  First genes:", df.index[:5].tolist())
except Exception as e:
    print("  FAILED:", e)

# Check H5 structure (preprocessing.py needs specific keys)
print("\nChecking H5 structure of human_matrix_v11.h5...")
try:
    s3 = s3fs.S3FileSystem(anon=True)
    with h5py.File(s3.open("mssm-seq-matrix/human_matrix_v11.h5", "rb"), "r") as f:
        def print_keys(name, obj):
            print(" ", name)
        f.visititems(print_keys)
except Exception as e:
    print("  FAILED:", e)
