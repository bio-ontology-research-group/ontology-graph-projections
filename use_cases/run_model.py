import sys
sys.path.append('../')

import mowl
mowl.init_jvm("10g")
import click as ck

from src.model import Model
from src.utils import seed_everything

@ck.command()
@ck.option('--use-case', '-case', required=True, type=ck.Choice(["go", "hpo"]))
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
@ck.option('--reduced_subsumption', '-rs', is_flag=True)
@ck.option('--reduced_existential', '-re', is_flag=True)
@ck.option('--test_file', '-tf', required=True, type=ck.Path(exists=True))
@ck.option('--device', '-d', required=True, type=ck.Choice(['cpu', 'cuda']))
@ck.option('--seed', '-s', required=True, type=int, default=42)
@ck.option("--only_test", '-ot', is_flag=True)
@ck.option('--result-dir', '-rd', required=True)
def main(use_case, graph_type, root, emb_dim, p_norm, margin, weight_decay, batch_size, lr, test_batch_size, num_negs, epochs,
         test_subsumption, test_existential,
         reduced_subsumption, reduced_existential, test_file, device, seed, only_test, result_dir):

    if not result_dir.endswith('.csv'):
        raise ValueError("For convenience, please specify a csv file as result_dir")
    
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
    print("\treduced_subsumption: ", reduced_subsumption)
    print("\treduced_existential: ", reduced_existential)
    print("\ttest_file: ", test_file)
    print("\tdevice: ", device)
    print("\tseed: ", seed)
    
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

    if graph_type == "rdf" and test_existential:
        mean_rank, mrr, hits_at_1, hits_at_10, hits_at_100 = model.test_rdf()
        with open(result_dir, "a") as f:
            line = f"{emb_dim},{margin},{weight_decay},{batch_size},{lr},{mean_rank},{mrr},{hits_at_1},{hits_at_10},{hits_at_100}\n"
            f.write(line)
        #mean_rank, mrr, hits_at_1, hits_at_10, hits_at_100 = model.test_filtered_rdf()
        #result_dir = result_dir.replace(".csv", "_filtered.csv")
        #with open(result_dir, "a") as f:
        #    line = f"{emb_dim},{margin},{weight_decay},{batch_size},{lr},{mean_rank},{mrr},{hits_at_1},{hits_at_10},{hits_at_100}\n"
        #    f.write(line)


    else:
        mean_rank, mrr, hits_at_1, hits_at_10, hits_at_100 = model.test()
        with open(result_dir, "a") as f:
            line = f"{emb_dim},{margin},{weight_decay},{batch_size},{lr},{mean_rank},{mrr},{hits_at_1},{hits_at_10},{hits_at_100}\n"
            f.write(line)
        mean_rank, mrr, hits_at_1, hits_at_10, hits_at_100 = model.test_filtered()
        result_dir = result_dir.replace(".csv", "_filtered.csv")
        with open(result_dir, "a") as f:
            line = f"{emb_dim},{margin},{weight_decay},{batch_size},{lr},{mean_rank},{mrr},{hits_at_1},{hits_at_10},{hits_at_100}\n"
            f.write(line)
        
if __name__ == "__main__":
    main()




 
