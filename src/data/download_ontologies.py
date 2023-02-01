import os
import wget

script_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(script_dir, "../../use_cases/")

def main():
    go_plus_path = "http://purl.obolibrary.org/obo/go/extensions/go-plus.owl"
    go_path = "http://purl.obolibrary.org/obo/go.owl"
    hpo_path = "http://purl.obolibrary.org/obo/hp.owl"

    go_plus_dir = os.path.join(root_dir, "go_plus", "data")
    go_dir = os.path.join(root_dir, "go", "data")
    hpo_dir = os.path.join(root_dir, "hpo", "data")

    if not os.path.exists(go_dir):
        print("Path does not exist: {}. Creating it...".format(go_dir))
        os.makedirs(go_dir)
    if not os.path.exists(go_plus_dir):
        print("Path does not exist: {}. Creating it...".format(go_plus_dir))
        os.makedirs(go_plus_dir)
    if not os.path.exists(hpo_dir):
        print("Path does not exist: {}. Creating it...".format(hpo_dir))
        os.makedirs(hpo_dir)

    wget.download(go_path, go_dir)
    print("GO downloaded into {}".format(go_dir))
    wget.download(go_plus_path, go_plus_dir)
    print("GO+ downloaded into {}".format(go_plus_dir))
    wget.download(hpo_path, hpo_dir)
    print("HPO downloaded into {}".format(hpo_dir))


if __name__ == "__main__":
    main()
    
