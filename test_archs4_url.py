import archs4py as a4

urls = [
    "https://s3.amazonaws.com/mssm-seq-matrix/human_gene_v2.5.h5",
    "https://mssm-seq-matrix.s3.amazonaws.com/human_gene_v2.5.h5",
    "https://s3.amazonaws.com/archs4-maayan/human_gene_v2.5.h5",
]

for url in urls:
    try:
        df = a4.data.rand_remote(url, 3, remove_sc=True)
        print("WORKS: " + url)
        print("Shape: " + str(df.shape))
        break
    except Exception as e:
        print("FAIL: " + url)
        print("  -> " + str(e))
