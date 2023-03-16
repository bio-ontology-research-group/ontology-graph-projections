# Projectors

To project an ontology, use the following command:

```
python [projector_name].py -i /path/to/ontology_name.owl
```
Projector names are:
 - owl2vec
 - onto2graph
 - rdf

The result will be a file `path/to/ontology_name.projector_name.edgelist`.

In the particular case of Onto2Graph, the command receives an extra parameter indicating the location of the Onto2Graph jar file:
```
python onto2graph_projector.py -i /path/to/ontology_name.owl -j /path/to/jar_file
 ```

You can check the `tests/tests_projectors.py` to see examples of how to call the projectors.
