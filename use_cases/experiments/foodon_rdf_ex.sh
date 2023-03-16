python ../run_model.py -case foodon -kge transe -g rdf -r ../foodon/data -dim 128 -m 0.0 -wd 0.000 -bs 4096 -lr 0.001 -tbs 8 -e 4000 -d cuda -rd result_foodon_rdf_ex_transe.csv -tf ../foodon/data/foodon_existential_subsumption_closure_filtered_non_trivial_no_leakage.csv -te -tbq

python ../run_model.py -case foodon -kge transr -g rdf -r ../foodon/data -dim 128 -m 0.0 -wd 0.000 -bs 4096 -lr 0.001 -tbs 8 -e 4000 -d cuda -rd result_foodon_rdf_ex_transr.csv -tf ../foodon/data/foodon_existential_subsumption_closure_filtered_non_trivial_no_leakage.csv -te -tbq

