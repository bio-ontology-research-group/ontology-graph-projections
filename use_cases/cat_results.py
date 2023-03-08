embed_dims = ["64", "128", "256"]
margins = ["0.0", "0.2", "0.4"]

def cat_results(directory, case, graph, test, reduced, both_quants):

    if case not in ["go", "foodon"]:
        raise ValueError("Case must be either 'go' or 'foodon'")
    if graph not in ["taxonomy", "owl2vec", "onto2graph", "rdf"]:
        raise ValueError("Graph must be either 'taxonomy', 'owl2vec', 'onto2graph' or 'rdf'")
    if test not in ["sub", "ex"]:
        raise ValueError("Test must be either 'sub' or 'ex'")
    if reduced not in ["True", "False"]:
        raise ValueError("Reduced must be either 'True' or 'False'")
    if both_quants not in ["True", "False"]:
        raise ValueError("Both quants must be either 'True' or 'False'")

    
    all_results = []
    for embed_dim in embed_dims:
        for margin in margins:
            filename = directory + "/" + case + "_" + graph + "_" + embed_dim + "_" + margin + "_" + test + ".csv"
            if reduced == "True":
                filename = filename.replace(".csv", "_red.csv")
            if both_quants == "True":
                filename = filename.replace(".csv", "_both_quants.csv")
                
            with open(filename, "r") as f:
                for line in f.readlines():
                    all_results.append(line)
    final_file = directory + "/" + case + "_" + graph + "_" + test + ".csv"
    if reduced == "True":
        final_file = final_file.replace(".csv", "_red.csv")
    if both_quants == "True":
        final_file = final_file.replace(".csv", "_both_quants.csv")
           
    with open(final_file, "w") as f:
        for res in all_results:
            f.write(f"{res}")


if __name__ == "__main__":
    import sys

    directory = sys.argv[1]
    case = sys.argv[2]
    graph = sys.argv[3]
    test = sys.argv[4]
    reduced = sys.argv[5]
    both_quants = sys.argv[6]
    cat_results(directory, case, graph, test, reduced, both_quants)
