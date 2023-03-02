import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
from mowl.projection import TaxonomyProjector
import click as ck

from tqdm import tqdm

def taxonomy_projector(input_ontology):
    ds = PathDataset(input_ontology)
    projector = TaxonomyProjector(bidirectional_taxonomy=True)
    graph = projector.project(ds.ontology)

    outfile = input_ontology.replace(".owl", ".taxonomy.edgelist")
    print(f"Graph computed. Writing into file: {outfile}")

    with open(outfile, "w") as f:
        for edge in tqdm(graph, total=len(graph)):
            f.write(f"{edge.src}\t{edge.rel}\t{edge.dst}\n")
    
    
@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
def main(input_ontology):
    taxonomy_projector(input_ontology)
    print("Done.")

if __name__ == '__main__':
    
    
    main()
