import mowl
mowl.init_jvm("10g")
import sys
import os
from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter
from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.formats import RDFXMLDocumentFormat



def main(input_file):
    ds = PathDataset(input_file)
    ontology = ds.ontology

    manager = OWLAPIAdapter().owl_manager
    # Save the ontology in RDF/XML format
    output_file = os.path.abspath(input_file)
    manager.saveOntology(ontology, RDFXMLDocumentFormat(), IRI.create("file:" + output_file))
    print("Saved ontology in RDF/XML format to %s" % input_file)


if __name__ == "__main__":
    input_file = sys.argv[1]
    main(input_file)
