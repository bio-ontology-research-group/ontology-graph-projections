python ../run_model.py -case go -g rdf -kge transe -r ../go/data -dim 64 -m 0.2 -wd 0.000 -bs 4096 -lr 0.01 -tbs 32 -e 4000 -d cuda -rd result_go_rdf_sub.csv -tf ../go/data/go_subsumption_closure_filtered.csv -ts 

python ../run_model.py -case go -g rdf -kge transe -r ../go/data -dim 64 -m 0.2 -wd 0.000 -bs 4096 -lr 0.01 -tbs 32 -e 4000 -d cuda -rd result_go_rdf_sub_red.csv -tf ../go/data/go_subsumption_closure_filtered.csv -ts -rs
