import rdflib
import time
import sys
import tqdm
import click as ck

def owl2rdf(owlfile):
    start_time = time.time()
    
    g = rdflib.Graph()
    g.parse (owlfile, format='application/rdf+xml')

    with open(owlfile.replace('.owl', '.rdf.edgelist'), 'w') as f:
        for s, p, o in tqdm.tqdm(g, total=len(g)):
            if isinstance(s, rdflib.term.Literal):
                continue
            if isinstance(o, rdflib.term.Literal):
                continue
                                                 
            f.write(str(s) + '\t' + str(p) + '\t' + str(o) + '\n')

    print("--- %s seconds ---" % (time.time() - start_time))


@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
def main(input_ontology):
    owlfile = input_ontology
    if not owlfile.endswith(".owl"):
        raise Exception("Input file must be an OWL file")

    owl2rdf(owlfile)


    
if __name__ == '__main__':
    
