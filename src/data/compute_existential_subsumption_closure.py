import mowl
mowl.init_jvm("20g")
import click as ck
from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from org.semanticweb.owlapi.model import ClassExpressionType as CT

import os
from tqdm import tqdm

@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
def main(input_ontology):
    """Computes the closure of axioms of the form C subclassof D, where C and D are both concept names"""

    manager = OWLAPIAdapter().owl_manager
    data_factory = manager.getOWLDataFactory()
    
    ds = PathDataset(input_ontology)
    ontology = ds.ontology
    classes = ontology.getClassesInSignature()
    relations = ontology.getObjectPropertiesInSignature()
    reasoner = ElkReasonerFactory().createReasoner(ontology)
    outfile_name = os.path.splitext(input_ontology)[0] + "_existential_subsumption_closure.csv"

    with open(outfile_name, "w") as outfile:
        for c in tqdm(classes):
            if c.getClassExpressionType() != CT.OWL_CLASS:
                continue
            for r in relations:
                some_values_from = data_factory.getOWLObjectSomeValuesFrom(r, c)
                subclasses = reasoner.getSubClasses(some_values_from, False)
                for subclass in subclasses.getFlattened():
                    c_str = str(c.toStringID())
                    r_str = str(r.toStringID())
                    subclass_str = str(subclass.toStringID())
                    outfile.write(f"{subclass_str},{r_str},{c_str}\n")
                             
    print(f"Done. Wrote to {outfile_name}")

if __name__ == "__main__":
    main()
