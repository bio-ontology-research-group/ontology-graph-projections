import rdflib
import time
import sys
import tqdm

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
    if not owlfile.endswith(".owl"):
        raise Exception("Input file must be an OWL file")

    owl2rdf(owlfile)
    
