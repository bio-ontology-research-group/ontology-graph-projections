python ../run_model.py -case foodon -kge transe -g owl2vec -r ../foodon/data -dim 256 -m 0.4 -wd 0.0001 -bs 16384 -lr 0.001 -tbs 8 -e 4000 -d cuda -rd result_foodon_owl2vec_ex_transe.csv -tf ../foodon/data/foodon_existential_subsumption_closure_filtered_non_trivial_no_leakage.csv -te -tbq


