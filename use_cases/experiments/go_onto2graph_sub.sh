python ../run_model.py -case go -g onto2graph -kge transe -r ../go/data -dim 128 -m 0.4 -wd 0.0005 -bs 4096 -lr 0.01 -tbs 32 -e 4000 -d cuda -rd result_go_onto2graph_sub.csv -tf ../go/data/go_subsumption_closure_filtered.csv -ts -ot

python ../run_model.py -case go -g onto2graph -kge transe -r ../go/data -dim 128 -m 0.4 -wd 0.0005 -bs 4096 -lr 0.01 -tbs 32 -e 4000 -d cuda -rd result_go_onto2graph_sub_red.csv -tf ../go/data/go_subsumption_closure_filtered.csv -ts -rs -ot
