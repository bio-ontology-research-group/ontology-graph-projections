python ../run_model.py -case go -g owl2vec -kge transe -r ../go/data -dim 128 -m 0.4 -wd 0.0 -bs 16384 -lr 0.001 -tbs 32 -e 4000 -d cuda -rd result_go_owl2vec_ex.csv -tf ../go/data/go_existential_subsumption_closure_filtered_non_trivial_no_leakage.csv -te -ot

python ../run_model.py -case go -g owl2vec -kge transe -r ../go/data -dim 128 -m 0.4 -wd 0.0 -bs 16384 -lr 0.001 -tbs 32 -e 4000 -d cuda -rd result_go_owl2vec_ex_red.csv -tf ../go/data/go_existential_subsumption_closure_filtered_non_trivial_no_leakage.csv -te -re -ot
