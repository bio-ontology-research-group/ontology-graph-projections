import mowl
mowl.init_jvm("20g")
import click as ck
from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter
import os
from tqdm import tqdm
import random
from org.semanticweb.owlapi.model import ClassExpressionType as CT
from org.semanticweb.owlapi.model import AxiomType, IRI
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.formats import RDFXMLDocumentFormat
from java.util import HashSet


@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
@ck.option("--percentage", "-p", type=float, default=0.1)
@ck.option("--subclass", "-s", is_flag=True)
@ck.option("--existential", "-e", is_flag=True)
def main(input_ontology, percentage, subclass, existential):
    """Remove axioms from an ontology. It will remove subclass axioms of
the form C subclassof D, or C subclassof some R. D. C and D are
concept names and R is a role. The percentage value indicates the
amount of axioms removed from the ontology. 
"""

    random.seed(42)
    manager = OWLAPIAdapter().owl_manager
    
    ds = PathDataset(input_ontology)
    ontology = ds.ontology
    remain_prc = int((1 - percentage)*100)
    subclass_remain = 100
    if subclass:
        subclass_remain = remain_prc

    existential_remain = 100
    if existential:
        existential_remain = remain_prc

    outfile_name = os.path.splitext(input_ontology)[0] + f"_{subclass_remain}_{existential_remain}.owl2"

    tbox_axioms = ontology.getTBoxAxioms(Imports.fromBoolean(True))

    if subclass:
        print("Removing subclass axioms of the form C subclassof D")
        all_subclass_axioms = []
        for axiom in tqdm(tbox_axioms, desc="Getting C subclassof D axioms"):
            if axiom.getAxiomType() != AxiomType.SUBCLASS_OF:
                continue
            if axiom.getSubClass().getClassExpressionType() != CT.OWL_CLASS:
                continue
            if axiom.getSuperClass().getClassExpressionType() != CT.OWL_CLASS:
                continue
            all_subclass_axioms.append(axiom)

            # shuffle the list of axioms
        all_subclass_axioms = list(all_subclass_axioms)
        random.shuffle(all_subclass_axioms)
        
        axioms_to_remove = all_subclass_axioms[:int(len(all_subclass_axioms)*percentage)]
        axioms_to_remove_j = HashSet()
        axioms_to_remove_j.addAll(axioms_to_remove)
        print(f"Removing {len(axioms_to_remove)} axioms from a total of {len(all_subclass_axioms)}")
        
        manager.removeAxioms(ontology, axioms_to_remove_j)
        
        
    if existential:
        print("Removing subclass axioms of the form C subclassof some R.D")
        relations_axioms = dict()

        relations = ontology.getObjectPropertiesInSignature(Imports.fromBoolean(True))
        for axiom in tqdm(tbox_axioms, desc="Getting C subclassof R.D axioms"):
            if axiom.getAxiomType() != AxiomType.SUBCLASS_OF:
                continue
            if axiom.getSubClass().getClassExpressionType() != CT.OWL_CLASS:
                continue
            if axiom.getSuperClass().getClassExpressionType() != CT.OBJECT_SOME_VALUES_FROM:
                continue
            filler = axiom.getSuperClass().getFiller()
            if filler.getClassExpressionType() != CT.OWL_CLASS:
                continue
            relation = axiom.getSuperClass().getProperty()
            relation_str = str(relation.toStringID())
            if relation_str not in relations_axioms:
                relations_axioms[relation_str] = []
            relations_axioms[relation_str].append(axiom)

        for rel_str, axioms in relations_axioms.items():
            num_axioms = len(axioms)
            random.shuffle(axioms)
            axioms_to_remove = axioms[:int(num_axioms*percentage)]
            axioms_to_remove_j = HashSet()
            axioms_to_remove_j.addAll(axioms_to_remove)
            print(f"Relation {rel_str}: Removing {len(axioms_to_remove)} axioms from a total of {num_axioms}")
            manager.removeAxioms(ontology, axioms_to_remove_j)
        
    
    outfile_name = os.path.abspath(outfile_name)
    manager.saveOntology(ontology, RDFXMLDocumentFormat(), IRI.create("file:" + outfile_name))
    
    
    print(f"Done. Wrote to {outfile_name}")

if __name__ == "__main__":
    main()
