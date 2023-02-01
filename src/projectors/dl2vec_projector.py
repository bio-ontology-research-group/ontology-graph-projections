import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
from mowl.projection import DL2VecProjector
import click as ck

from tqdm import tqdm

@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
def main(input_ontology):
    
    ds = PathDataset(input_ontology)
    projector = DL2VecProjector(bidirectional_taxonomy=True)
    graph = projector.project(ds.ontology)

    outfile = input_ontology.replace(".owl", ".dl2vec.edgelist")
    print(f"Graph computed. Writing into file: {outfile}")

    with open(outfile, "w") as f:
        for edge in tqdm(graph, total=len(graph)):
            f.write(f"{edge.src}\t{edge.rel}\t{edge.dst}\n")
    print("Done.")

if __name__ == '__main__':

    main()
