import sys
sys.path.append('../')

import mowl
mowl.init_jvm("10g")
import click as ck
import os
from src.model import Model
from src.utils import seed_everything
import gc
import torch as th

@ck.command()
@ck.option('--use-case', '-case', required=True, type=ck.Choice(["foodon", "go", "hpo", "go_link_pred", "foodon_link_pred"]))
@ck.option('--graph-type', '-g', required=True, type=ck.Choice(['rdf', "owl2vec", "taxonomy", "dl2vec", 'onto2graph']))
@ck.option('--root', '-r', required=True, type=ck.Path(exists=True))
@ck.option('--emb-dim', '-dim', required=True, type=int, default=256)
@ck.option('--p-norm', '-norm' , required=True, type=int, default=2)
@ck.option('--margin', '-m', required=True, type=float, default=0.1)
@ck.option('--weight-decay', '-wd', required=True, type=float, default = 0.0)
@ck.option('--batch-size', '-bs', required=True, type=int, default=4096*8)
@ck.option('--lr', '-lr', required=True, type=float, default=0.001)
@ck.option('--test-batch-size', '-tbs', required=True, type=int, default=32)
@ck.option('--num-negs', '-negs', required=True, type=int, default=2)
@ck.option('--epochs', '-e', required=True, type=int, default=300)
@ck.option('--test-subsumption', '-ts', is_flag=True)
@ck.option('--test-existential', '-te', is_flag=True)
@ck.option('--test-both-quantifiers', '-tbq', is_flag=True)
@ck.option('--reduced_subsumption', '-rs', is_flag=True)
@ck.option('--reduced_existential', '-re', is_flag=True)
@ck.option('--test_file', '-tf', required=True, type=ck.Path(exists=True))
@ck.option('--device', '-d', required=True, type=ck.Choice(['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']))
@ck.option('--seed', '-s', required=True, type=int, default=42)
@ck.option("--only_train", '-otr', is_flag=True)
@ck.option("--only_test", '-ot', is_flag=True)
@ck.option('--result-dir', '-rd', required=True)
def main(use_case, graph_type, root, emb_dim, p_norm, margin, weight_decay, batch_size, lr, test_batch_size, num_negs, epochs,
         test_subsumption, test_existential,
         test_both_quantifiers,
         reduced_subsumption, reduced_existential, test_file, device, seed, only_train, only_test, result_dir):

    if not result_dir.endswith('.csv'):
        raise ValueError("For convenience, please specify a csv file as result_dir")

    if root.endswith('/'):
        root = root[:-1]

    #get parent of root
    root_parent = os.path.dirname(root)
        
    models_dir = os.path.join(root_parent, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
        
    print("Configuration:")
    print("\tuse_case: ", use_case)
    print("\tgraph_type: ", graph_type)
    print("\troot: ", root)
    print("\temb_dim: ", emb_dim)
    print("\tp_norm: ", p_norm)
    print("\tmargin: ", margin)
    print("\tweight_decay: ", weight_decay)
    print("\tbatch_size: ", batch_size)
    print("\tlr: ", lr)
    print("\ttest_batch_size: ", test_batch_size)
    print("\tnum_negs: ", num_negs)
    print("\tepochs: ", epochs)
    print("\ttest_subsumption: ", test_subsumption)
    print("\ttest_existential: ", test_existential)
    print("\ttest_both_quantifiers: ", test_both_quantifiers)
    print("\treduced_subsumption: ", reduced_subsumption)
    print("\treduced_existential: ", reduced_existential)
    print("\ttest_file: ", test_file)
    print("\tdevice: ", device)
    print("\tseed: ", seed)
    print("\tonly_train: ", only_train)
    print("\tonly_test: ", only_test)
    
    seed_everything(seed)
    
    model = Model(use_case,
                  graph_type,
                  root,
                  emb_dim = emb_dim,
                  p_norm = p_norm,
                  margin = margin,
                  weight_decay = weight_decay,
                  batch_size = batch_size,
                  lr = lr,
                  test_batch_size = test_batch_size,
                  num_negs = num_negs,
                  reduced_subsumption = reduced_subsumption,
                  reduced_existential = reduced_existential,
                  test_file = test_file,
                  epochs = epochs,
                  test_subsumption = test_subsumption,
                  test_existential = test_existential,
                  device = device,
                  seed = seed)

    if not only_test:
        model.train()

    if not only_train:
        assert os.path.exists(test_file)
        params = (emb_dim, margin, weight_decay, batch_size, lr)

        if test_subsumption:
            print("Start testing subsumption")
            raw_metrics, filtered_metrics = model.test(False, False)
            save_results(params, raw_metrics, filtered_metrics, result_dir)
                                                                                                                
        if test_existential:
            print("Start testing existential")
            if not test_both_quantifiers:
                raw_metrics, filtered_metrics = model.test(True, False)
                save_results(params, raw_metrics, filtered_metrics, result_dir)

            else:
                result_dir_both_quants = result_dir.replace('.csv', '_both_quants.csv')
                raw_metrics, filtered_metrics = model.test(True, True)
                save_results(params, raw_metrics, filtered_metrics, result_dir_both_quants)
                                                                                                                

def save_results(params, raw_metrics, filtered_metrics, result_dir):
    emb_dim, margin, weight_decay, batch_size, lr = params
    mr, mrr, h1, h10, h100, auc = raw_metrics
    mr_f, mrr_f, h1_f, h10_f, h100_f, auc_f = filtered_metrics
    with open(result_dir, 'a') as f:
        line = f"{emb_dim},{margin},{weight_decay},{batch_size},{lr},{mr},{mrr},{h1},{h10},{h100},{auc},{mr_f},{mrr_f},{h1_f},{h10_f},{h100_f},{auc_f}\n"
        f.write(line)
        
if __name__ == "__main__":
    main()




 
