set -e

python ../run_model.py -case foodon -kge transe -g onto2graph -r ../foodon/data -dim 256 -m 0.0 -wd 0.000 -bs 16384 -lr 0.01 -tbs 32 -e 4000 -d cuda -rd result_foodon_onto2graph_ex_transe.csv -tf ../foodon/data/foodon_existential_subsumption_closure_filtered_non_trivial.csv -te

python ../run_model.py -case foodon -kge transe -g onto2graph -r ../foodon/data -dim 256 -m 0.0 -wd 0.000 -bs 16384 -lr 0.01 -tbs 32 -e 4000 -d cuda -rd result_foodon_onto2graph_ex_transe_bq.csv -tf ../foodon/data/foodon_existential_subsumption_closure_filtered_non_trivial.csv -te -tbq -ot

python ../run_model.py -case foodon -kge transd -g onto2graph -r ../foodon/data -dim 256 -m 0.0 -wd 0.000 -bs 16384 -lr 0.01 -tbs 32 -e 4000 -d cuda -rd result_foodon_onto2graph_ex_transd.csv -tf ../foodon/data/foodon_existential_subsumption_closure_filtered_non_trivial.csv -te

python ../run_model.py -case foodon -kge transd -g onto2graph -r ../foodon/data -dim 256 -m 0.0 -wd 0.000 -bs 16384 -lr 0.01 -tbs 32 -e 4000 -d cuda -rd result_foodon_onto2graph_ex_transd_bq.csv -tf ../foodon/data/foodon_existential_subsumption_closure_filtered_non_trivial.csv -te -tbq -ot
