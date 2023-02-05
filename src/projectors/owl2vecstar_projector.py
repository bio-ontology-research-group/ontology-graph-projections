"""Generates a graph from ontology using OWL2VecStar projection rules"""
import sys
from owl2vec_star.lib.Onto_Projection import Reasoner, OntologyProjection
import rdflib

def owl2vecstar_projection(ontology_file):

    projection = OntologyProjection(ontology_file, reasoner=Reasoner.STRUCTURAL, only_taxonomy=False,
                                    bidirectional_taxonomy=True, include_literals=False, avoid_properties=set(),
                                    additional_preferred_labels_annotations=set(),
                                    additional_synonyms_annotations=set(),
                                    memory_reasoner='13351')

    projection.extractProjection()
    output_file = ontology_file.replace('.owl', '.owl2vec.ttl')
    projection.saveProjectionGraph(output_file)

    g = rdflib.Graph()
    g.parse(output_file, format='turtle')

    with open(output_file.replace('.ttl', '.edgelist'), 'w') as f:
        for s, p, o in g:
            if isinstance(s, rdflib.term.Literal):
                continue
            if isinstance(o, rdflib.term.Literal):
                continue
            #if " " in s or " " in o:
            #    continue
            #if "http://langual" in s or "http://langual" in o:
            #    continue
            #if "oboInOwl" in p or "annotated" in p or "label" in p:
            #    continue
            #if not s.startswith("http") and not len(s) > 20:
            #    continue
            #if not o.startswith("http") and not len(o) > 20:
            #    continue
            f.write(str(s) + '\t' + str(p) + '\t' + str(o) + '\n')


if __name__ == '__main__':
    ontology_file = sys.argv[1]
    owl2vecstar_projection(ontology_file)
    print("Done!")
