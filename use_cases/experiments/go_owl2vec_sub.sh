python ../run_model.py -case go -g owl2vec -kge transe -r ../go/data -dim 128 -m 0.2 -wd 0.0001 -bs 8192 -lr 0.01 -tbs 32 -e 4000 -d cuda -rd result_go_owl2vec_sub.csv -tf ../go/data/go_subsumption_closure_filtered.csv -ts -ot

python ../run_model.py -case go -g owl2vec -kge transe -r ../go/data -dim 128 -m 0.2 -wd 0.0001 -bs 8192 -lr 0.01 -tbs 32 -e 4000 -d cuda -rd result_go_owl2vec_sub_red.csv -tf ../go/data/go_subsumption_closure_filtered.csv -ts -rs -ot
