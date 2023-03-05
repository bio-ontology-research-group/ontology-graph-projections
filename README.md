# Ontologies as graphs: an analysis on the properties of graph projections on axiom inference


## Method and results
	
## Repository overview

Provide an overview of the directory structure and files, for example:

```
├── README.md
├── environment.yml
├── src
│   ├── data
│   ├── projectors
└── use_cases
    ├── foodon
    │   ├── data
    │   └── models
    └── go
        ├── data
        └── models
 
```

## Running instructions

### Dependencies
- Python 3.8
- Anaconda

### Set up environment

```bash
git clone --recursive https://github.com/bio-ontology-research-group/ontology_projections.git

cd ontology-projections
conda env create -f environment.yml
conda activate projections

cd use_cases
```

## Running the model

To run the script, use the ``run_model.py`` script. The parameters are the following:

| Parameter             | Command line | Options                                                                               |
|-----------------------|--------------|---------------------------------------------------------------------------------------|
| case                  | -case        | go, foodon                                                                            |
| graph                 | -g           | taxonomy, owl2vec, onto2graph, rdf                                                    |
| root directory        | -r           | [case]/data                                                                           |
| embedding dimension   | -dim         | 64, 128, 256                                                                          |
| margin                | -margin      | 0.0, 0.2, 0.4                                                                         |
| weight decay          | -wd          | 0.0000, 0.0001, 0.0005                                                                |
| batch size            | -bs          | 4096, 8192, 16834                                                                     |
| learning rate         | -lr          | 0.1, 0.01, 0.001                                                                      |
| testing batch size    | -tbs         | 8, 16                                                                                 |
| epochs                | -e           | 1000, 2000, ...                                                                       |
| device                | -d           | cpu, cuda                                                                             |
| reduced subsumption   | -rs          | Train on ontology with some axioms $C \sqsubseteq D$ removed                          |
| reduced existential   | -re          | Train on ontology with some axioms $C \sqsubseteq \exists R.D$ removed                |
| results file          | -rd          | name\_of\_results\_csv\_file.csv                                                      |
| testing file          | -tf          | name\_of\_testing\_file.csv                                                           |
| test subsumption      | -ts          | Flag to test $C \sqsubseteq D$ axioms                                                 |
| test existential      | -te          | Flag to test $C \sqsubseteq \exists R.D$ axioms                                       |
| test both quantifiers | -tbq         | Flag to test axioms $C \sqsubseteq \exists R.D$ including $C \sqsubseteq \forall R.D$ |
| only train            | -otr         | Flag to only perform training                                                         |
| only test             | -ot          | Flag to only perform testing                                                          |

For example, to train the Gene Ontology reduced version with the taxonomy projection, we can run
```
python run_model.py -case go -g taxonomy -r go/data -dim 64 -m 0.2 -wd 0.0001 -bs 4096 -lr 0.1 -tbs 8 -e 4000 -d cuda -rd results_tax_sub.csv -tf foodon/data/go_subsumption_closure_filtered.csv -ts -rs
```

## More resources


## About


