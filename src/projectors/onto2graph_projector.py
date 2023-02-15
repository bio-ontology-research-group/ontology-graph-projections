import rdflib
import time
import sys
import tqdm
import subprocess
import os

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

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    owlfile = sys.argv[1]
    dummy = sys.argv[2]
    
    if not owlfile.endswith('.owl'):
        raise Exception('File must be an OWL file')

    rdfxmlfile = owlfile.replace('.owl', '.onto2graph')

    if dummy == "True":
        print("Dummy mode")
        command = ['java', '-jar', 'Onto2Graph/target/Onto2Graph-1.0.jar', '-ont', owlfile, '-out', rdfxmlfile, "-r", "STRUCTURAL", '-f', 'RDFXML', '-nt', '8']
    else:
        command = ['java', '-jar', 'Onto2Graph/target/Onto2Graph-1.0.jar', '-ont', owlfile, '-out', rdfxmlfile, '-eq', "true", "-op", "[*]", '-r', 'ELK', '-f', 'RDFXML', '-nt', '8']
    
    rdfxmlfile = rdfxmlfile + '.rdfxml'

    print("Running Onto2Graph")
    result = subprocess.run(command, stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
    print("Onto2Graph finished")
    print("Converting to edgelist")
    owl2rdf(rdfxmlfile)
    
