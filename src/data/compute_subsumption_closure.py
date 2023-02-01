import mowl
mowl.init_jvm("20g")
import click as ck
from mowl.datasets import PathDataset
from org.semanticweb.elk.owlapi import ElkReasonerFactory
import os
from tqdm import tqdm

@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
def main(input_ontology):
    """Computes the closure of axioms of the form C subclassof D, where C and D are both concept names"""

    ds = PathDataset(input_ontology)
    ontology = ds.ontology
    classes = ontology.getClassesInSignature()

    reasoner = ElkReasonerFactory().createReasoner(ontology)
    outfile_name = os.path.splitext(input_ontology)[0] + "_subsumption_closure.csv"

    with open(outfile_name, "w") as outfile:
        for c in tqdm(classes):
            superclasses = reasoner.getSuperClasses(c, False)
            for d in superclasses.getFlattened():
                c_str = str(c.toStringID())
                d_str = str(d.toStringID())
                outfile.write(f"{c_str},{d_str}\n")

    print(f"Done. Wrote to {outfile_name}")

if __name__ == "__main__":
    main()
