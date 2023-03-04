import rdflib
import time
import sys
import tqdm
import subprocess
import os
import click as ck

def owl2rdf(owlfile):
    start_time = time.time()
    
    g = rdflib.Graph()
    g.parse (owlfile, format='application/rdf+xml')

    with open(owlfile.replace('.rdfxml', '.edgelist'), 'w') as f:
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
    rdfxmlfile = input_ontology.replace('.owl', '.onto2graph')

    command = ['java', '-jar', 'Onto2Graph/target/Onto2Graph-1.0.jar', '-ont', input_ontology, '-out', rdfxmlfile, '-eq', "true", "-op", "[*]", '-r', 'ELK', '-f', 'RDFXML', '-nt', '8']
    
    rdfxmlfile = rdfxmlfile + '.rdfxml'

    print("Running Onto2Graph")
    result = subprocess.run(command, stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
    print("Onto2Graph finished")
    print("Converting to edgelist")
    owl2rdf(rdfxmlfile)
                    
if __name__ == '__main__':
    main()
