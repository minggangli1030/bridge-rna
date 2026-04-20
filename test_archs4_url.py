import s3fs

# List contents of the mssm-seq-matrix bucket to find correct file paths
print("Listing mssm-seq-matrix bucket...")
try:
    s3 = s3fs.S3FileSystem(anon=True)
    files = s3.ls("mssm-seq-matrix/")
    for f in files:
        print(" ", f)
except Exception as e:
    print("mssm-seq-matrix failed:", e)

# Also try maayanlab-public bucket
print("\nListing maayanlab-public bucket...")
try:
    s3 = s3fs.S3FileSystem(anon=True)
    files = s3.ls("maayanlab-public/")
    for f in files:
        print(" ", f)
except Exception as e:
    print("maayanlab-public failed:", e)
