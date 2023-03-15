import pandas as pd
import sys

filename = sys.argv[1]

if not "edgelist" in filename:
    print("Not an edgelist, exiting")
    sys.exit(1)

df = pd.read_csv(filename, sep="\t", header=None)
df.columns = ["head", "rel", "tail"]

df = df[df["head"] != df["tail"]]

outname = filename.replace(".edgelist", "_no_identity.edgelist")
df.to_csv(outname, sep="\t", header=False, index=False)

