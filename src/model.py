import os
import pandas as pd
import numpy as np
from src.utils import FastTensorDataLoader

import torch as th
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn

from pykeen.models import TransE
from pykeen.triples import TriplesFactory

from tqdm import trange, tqdm

from mowl.owlapi.defaults import BOT, TOP
import logging
logging.basicConfig(level=logging.DEBUG)

prefix = {
    "go": "go",
    "foodon": "foodon",
}
suffix = {
    "taxonomy": "taxonomy.edgelist",
    "dl2vec": "dl2vec.edgelist",
    "onto2graph": "onto2graph.edgelist",
    "owl2vec": "owl2vec.edgelist",
    "rdf": "rdf.edgelist",
}

rel_name = {
    "taxonomy": "http://subclassof",
    "dl2vec": "http://subclassof",
    "onto2graph": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "owl2vec": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "rdf": "http://www.w3.org/2000/01/rdf-schema#subClassOf"
}


class ProjectionModule(nn.Module):
    def __init__(self, triples_factory, embedding_dim, random_seed):
        super().__init__()
        self.triples_factory = triples_factory
        self.embedding_dim = embedding_dim
        self.random_seed = random_seed
        self.kg_module =  TransE(triples_factory=self.triples_factory,
                                 embedding_dim=self.embedding_dim,
                                 scoring_fct_norm=2, #self.p_norm, # Trans[E,R]
                                 random_seed = self.random_seed)
                        
    def forward(self, data):
        h, r, t = data
        x = -self.kg_module.forward(h, r, t, mode=None)
        assert (x>=0).all()
        return x
        
class Model():
    def __init__(self,
                 use_case,
                 graph_type,
                 root,
                 emb_dim = 32,
                 p_norm = 1,
                 margin = 1,
                 weight_decay = 0,
                 batch_size = 128,
                 lr = 0.001,
                 test_batch_size = 8,
                 num_negs = 1,
                 epochs = 10,
                 test_subsumption = False,
                 test_existential = False,
                 reduced_subsumption = False,
                 reduced_existential = False,
                 test_file=None,
                 device = "cpu",
                 seed = 42
                 ):

        if not test_subsumption and not test_existential:
            raise ValueError("At least one of test_subsumption or test_existential must be True")
        if test_subsumption and test_existential:
            raise ValueError("Only one of test_subsumption or test_existential can be True")
        
        self.use_case = use_case
        self.graph_type = graph_type
        self.root = root
        self.emb_dim = emb_dim
        self.p_norm = p_norm
        self.margin = margin
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr = lr
        self.test_batch_size = test_batch_size
        self.num_negs = num_negs
        self.epochs = epochs
        self.test_subsumption = test_subsumption
        self.test_existential = test_existential
        self.reduced_subsumption = reduced_subsumption
        self.reduced_existential = reduced_existential
        self.test_file = test_file
        self.device = device
        self.seed = seed
        self.initial_tolerance = 10
        
        self._graph = None
        self._class_to_id = None
        self._relation_to_id = None
        
        self._inferred_ancestors = None
        self._inferred_ancestors_tensor = None
        self._train_tuples = None
        self._valid_tuples = None
        self._test_tuples = None
        self._model_path = None
        self._graph_path = None
        self._ontology_classes = None
        self._ontology_classes_idxs = None
        self._ontology_relations = None
        self._ontology_relations_idxs = None
        
        self._triples_factory = None

        self.model = ProjectionModule(triples_factory=self.triples_factory,
                                      embedding_dim=self.emb_dim,
                                      random_seed = self.seed
                                      )
                                                        
        assert os.path.exists(self.root), f"Root directory '{self.root}' does not exist"

    
    @property
    def graph(self):
        if self._graph is not None:
            return self._graph
        
        print(f"Loading graph from {self.graph_path}")
        graph = pd.read_csv(self.graph_path, sep="\t", header=None)
        graph.columns = ["head", "relation", "tail"]
        self._graph = graph
        print("Done")
        return self._graph
    

    @property
    def triples_factory(self):
        if self._triples_factory is not None:
            return self._triples_factory

        tensor = []
        for row in self.graph.itertuples():
            tensor.append([self.class_to_id[row.head],
                           self.relation_to_id[row.relation],
                           self.class_to_id[row.tail]])

        tensor = th.LongTensor(tensor)
        self._triples_factory = TriplesFactory(tensor, self.class_to_id, self.relation_to_id, create_inverse_triples=True)
        return self._triples_factory
        
        
    @property
    def class_to_id(self):
        if self._class_to_id is not None:
            return self._class_to_id

        graph_classes = set(self.graph["head"].unique()) | set(self.graph["tail"].unique())
        graph_classes.add(BOT)
        graph_classes.add(TOP)
        ont_classes = set(self.ontology_classes)
        all_classes = list(graph_classes | ont_classes)
        all_classes.sort()
        self._class_to_id = {c: i for i, c in enumerate(all_classes)}
        logging.info(f"Number of graph nodes: {len(self._class_to_id)}")
        return self._class_to_id

    @property
    def id_to_class(self):
        return {v: k for k, v in self.class_to_id.items()}
    
    @property
    def relation_to_id(self):
        if self._relation_to_id is not None:
            return self._relation_to_id

        graph_rels = list(self.graph["relation"].unique())
        graph_rels.sort()
        self._relation_to_id = {r: i for i, r in enumerate(graph_rels)}
        logging.info(f"Number of graph relations: {len(self._relation_to_id)}")
        return self._relation_to_id

    @property
    def id_to_relation(self):
        return {v: k for k, v in self.relation_to_id.items()}

    @property
    def graph_path(self):
        if self._graph_path is not None:
            return self._graph_path

        graph_name = prefix[self.use_case]

        if self.test_subsumption:
            if self.reduced_subsumption:
                graph_name += "_90_100"
            if self.reduced_existential:
                graph_name += "_100_90"
        elif self.test_existential:
            if self.reduced_existential:
                graph_name += "_100_90"
            if self.reduced_subsumption:
                graph_name += "_90_100"
        else:
            raise ValueError("At least one of test_subsumption or test_existential must be True")
        
        graph_path = os.path.join(self.root, f"{graph_name}.{suffix[self.graph_type]}")
        assert os.path.exists(graph_path), f"Graph file {graph_path} does not exist"
        self._graph_path = graph_path
        return self._graph_path
        
    @property
    def model_path(self):
        if self._model_path is not None:
            return self._model_path

        params_str = f"{self.graph_type}"
        params_str += f".{self.reduced_subsumption}"
        params_str += f".{self.reduced_existential}"
        params_str += f"_dim{self.emb_dim}"
        params_str += f"_norm{self.p_norm}"
        params_str += f"_marg{self.margin}"
        params_str += f"_reg{self.weight_decay}"
        params_str += f"_bs{self.batch_size}"
        params_str += f"_lr{self.lr}"
        params_str += f"_negs{self.num_negs}"
        
        models_dir = os.path.dirname(self.root)
        models_dir = os.path.join(models_dir, "models")

        basename = f"{params_str}.model.pt"
        self._model_path = os.path.join(models_dir, basename)
        return self._model_path

    @property
    def test_tuples_path(self):
        path = self.test_file
        assert os.path.exists(path), f"Test tuples file {path} does not exist"
        return path

    @property
    def classes_path(self):
        path = os.path.join(self.root, "classes.txt")
        assert os.path.exists(path), f"Classes file {path} does not exist"
        return path

    @property
    def relations_path(self):
        path = os.path.join(self.root, "relations.txt")
        assert os.path.exists(path), f"Relations file {path} does not exist"
        return path

    @property
    def ontology_classes(self):
        if self._ontology_classes is not None:
            return self._ontology_classes
        
        eval_classes = pd.read_csv(self.classes_path, sep=",", header=None)
        eval_classes.columns = ["classes"]
        eval_classes = eval_classes["classes"].values
        eval_classes.sort()

        self._ontology_classes = eval_classes
        return self._ontology_classes
    
    @property
    def ontology_classes_idxs(self):
        if self._ontology_classes_idxs is not None:
            return self._ontology_classes_idxs
        
        eval_classes = pd.read_csv(self.classes_path, sep=",", header=None)
        eval_classes.columns = ["classes"]
        eval_classes = eval_classes["classes"].values.tolist()
        eval_classes.append(BOT)
        eval_classes.append(TOP)
        eval_classes.sort()

        eval_class_to_id = {c: self.class_to_id[c] for c in eval_classes}
        ontology_classes_idxs = th.tensor(list(eval_class_to_id.values()), dtype=th.long, device=self.device)
        self._ontology_classes_idxs = ontology_classes_idxs
        return self._ontology_classes_idxs

    @property
    def ontology_relations(self):
        if self._ontology_relations is not None:
            return self._ontology_relations
        
        eval_relations = pd.read_csv(self.relations_path, sep=",", header=None)
        eval_relations.columns = ["relations"]
        eval_relations = eval_relations["relations"].values
        eval_relations.sort()

        self._ontology_relations = eval_relations
        return self._ontology_relations
    
    @property
    def ontology_relations_idxs(self):
        if self._ontology_relations_idxs is not None:
            return self._ontology_relations_idxs
        
        eval_relations = pd.read_csv(self.relations_path, sep=",", header=None)
        eval_relations.columns = ["relations"]
        eval_relations = eval_relations["relations"].values.tolist()
        eval_relations.sort()

        if self.graph_type == "rdf":
            eval_rel_to_id = {c: self.class_to_id[c] for c in eval_relations if c in self.class_to_id}
        else:
            eval_rel_to_id = {c: self.relation_to_id[c] for c in eval_relations if c in self.relation_to_id}
        
        logging.info(f"Number of ontology relations found in projection: {len(eval_rel_to_id)}")
        ontology_relations_idxs = th.tensor(list(eval_rel_to_id.values()), dtype=th.long, device=self.device)
        self._ontology_relations_idxs = ontology_relations_idxs
        return self._ontology_relations_idxs

    
    def create_graph_train_dataloader(self):
        heads = [self.class_to_id[h] for h in self.graph["head"]]
        rels = [self.relation_to_id[r] for r in self.graph["relation"]]
        tails = [self.class_to_id[t] for t in self.graph["tail"]]

        heads = th.LongTensor(heads)
        rels = th.LongTensor(rels)
        tails = th.LongTensor(tails)
        
        dataloader = FastTensorDataLoader(heads, rels, tails,
                                          batch_size=self.batch_size, shuffle=True)
        return dataloader
            

    def create_subsumption_dataloader(self, tuples_path, batch_size):
        tuples = pd.read_csv(tuples_path, sep=",", header=None)
        num_cols = tuples.shape[1]
        if num_cols == 2:
            tuples.columns = ["head", "tail"]
        elif num_cols == 3:
            tuples.columns = ["head", "relation", "tail"]
        else:
            raise ValueError(f"Invalid number of columns in {tuples_path}")

        heads = [self.class_to_id[h] for h in tuples["head"]]
        tails = [self.class_to_id[t] for t in tuples["tail"]]

        heads = th.tensor(heads, dtype=th.long)
        tails = th.tensor(tails, dtype=th.long)
        
        if num_cols == 2:
            rel_idx = self.relation_to_id[rel_name[self.graph_type]]
            rels = rel_idx * th.ones_like(heads)
        else:
            print(self.relation_to_id)
            if self.graph_type == "rdf":
                rels = [self.class_to_id[r] for r in tuples["relation"]]
            else:
                rels = [self.relation_to_id[r] for r in tuples["relation"]]
            rels = th.tensor(rels, dtype=th.long)
        
        dataloader = FastTensorDataLoader(heads, rels, tails, batch_size=batch_size, shuffle=True)
        return dataloader

    def prediction_dataloader(self):
        dataloader = FastTensorDataLoader(self.ontology_classes_idxs, batch_size = self.test_batch_size, shuffle=False)
        return dataloader


    
    
    
    def train(self):

        optimizer = optim.Adam(self.model.parameters(), lr=0.000001, weight_decay = self.weight_decay)
        min_lr = self.lr/100  #0.0001 #0.000001
        max_lr = self.lr #0.01 #0.0001
        scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up = 30, cycle_momentum = False)
                        
        self.model = self.model.to(self.device)

        graph_dataloader = self.create_graph_train_dataloader()

        tolerance = 0
        best_loss = float("inf")
        ont_classes_idxs = th.tensor(list(self.ontology_classes_idxs), dtype=th.long, device=self.device)

        for epoch in trange(self.epochs, desc=f"Training..."):
            print("Epoch: ", epoch+1)

            self.model.train()

            graph_loss = 0
            for head, rel, tail in tqdm(graph_dataloader, desc="Processing batches"):
                head = head.to(self.device)
                rel = rel.to(self.device)
                tail = tail.to(self.device)
                
                data = (head, rel, tail)
                pos_logits = self.model.forward(data)

                neg_logits = 0
                for i in range(self.num_negs):
                    neg_tail = th.randint(0, len(self.class_to_id), (len(head),), device=self.device)
                    data = (head, rel, neg_tail)
                    neg_logits += self.model.forward(data)
                neg_logits /= self.num_negs
                
                batch_loss= th.relu(pos_logits - neg_logits + self.margin).mean()

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                graph_loss += batch_loss.item()

            graph_loss /= len(graph_dataloader)

            if best_loss > graph_loss:
                best_loss = graph_loss
                th.save(self.model.state_dict(), self.model_path)
                tolerance = self.initial_tolerance
                print("Model saved")
            else:
                tolerance -= 1
                if tolerance == 0:
                    print("Early stopping")
                    break

            print(f"Train loss: {graph_loss:.6f}\n")
                

    def rdf_existential_forward(self, heads_idxs, rel_idxs, tails_idxs, both_quantifiers):
        somevaluesfrom_relation_id = self.relation_to_id['http://www.w3.org/2002/07/owl#someValuesFrom']
        somevaluesfrom = somevaluesfrom_relation_id * th.ones_like(heads_idxs)
                
        data1_ex = (heads_idxs, somevaluesfrom, tails_idxs)
        logits1_ex = self.model.forward(data1_ex)
        logits1_ex = logits1_ex.reshape(-1, len(self.ontology_classes_idxs))

        data2_ex = (rel_idxs, somevaluesfrom, tails_idxs)
        logits2_ex = self.model.forward(data2_ex)
        logits2_ex = logits2_ex.reshape(-1, len(self.ontology_classes_idxs))

        logits_ex = logits1_ex + logits2_ex

        if both_quantifiers:
            
            allvaluesfrom_relation_id = self.relation_to_id['http://www.w3.org/2002/07/owl#allValuesFrom']
            allvaluesfrom = allvaluesfrom_relation_id * th.ones_like(heads_idxs)

            data1_all = (heads_idxs, allvaluesfrom, tails_idxs)
            logits1_all = self.model.forward(data1_all)
            logits1_all = logits1_all.reshape(-1, len(self.ontology_classes_idxs))

            data2_all = (rel_idxs, allvaluesfrom, tails_idxs)
            logits2_all = self.model.forward(data2_all)
            logits2_all = logits2_all.reshape(-1, len(self.ontology_classes_idxs))

            logits_al = logits1_all + logits2_all
            logits = th.cat((logits_ex, logits_al), dim=1)
        else:
            logits = logits_ex

        return logits

    def existential_forward(self, head_idxs, rel_idxs, tail_idxs, both_quantifiers):
        logits = self.model.forward((head_idxs, rel_idxs, tail_idxs))
        logits = logits.reshape(-1, len(self.ontology_classes_idxs))
        if both_quantifiers:
            logits = th.cat([logits, logits], dim=1)
        return logits

          
    def normal_forward(self, head_idxs, rel_idxs, tail_idxs):
          logits = self.model.forward((head_idxs, rel_idxs, tail_idxs))
          logits = logits.reshape(-1, len(self.ontology_classes_idxs))
          return logits

    def get_preds_and_labels(self, existential_axioms = False, both_quantifiers = False):
                                                    
        logging.info("Getting predictions and labels")

        num_testing_heads = len(self.ontology_classes_idxs)
        num_testing_tails = num_testing_heads
        
        subsumption_relation = rel_name[self.graph_type]
        rel_to_eval_id = {subsumption_relation: self.relation_to_id[subsumption_relation]}

        
        self.eval_relations = {subsumption_relation: 0} # this variable is defined here for the first time and it is used later in compute_ranking_metrics function
        
        if existential_axioms:
            num_relations = len(self.ontology_relations_idxs)
            assert num_relations > 1, f"Number of relations: {num_relations}"

            rel_to_eval_id = {rel: idx for idx, rel in enumerate(self.ontology_relations)}
                                                        
            self.eval_relations = dict(zip(self.ontology_relations, range(num_relations)))
            if  both_quantifiers:
                num_testing_tails *= 2

        num_relations = len(rel_to_eval_id)
        num_eval_relations = len(self.eval_relations)
        logging.debug(f"num_testing_heads: {num_testing_heads}")
        logging.debug(f"num_testing_tails: {num_testing_tails}")
        logging.debug(f"num_relations: {num_relations}")
        logging.debug(f"num_eval_relations: {num_eval_relations}")

            
        preds = -1 * np.ones((num_eval_relations, num_testing_heads, num_testing_tails))
        trlabels = np.ones((num_eval_relations, num_testing_heads, num_testing_tails))

        logging.debug(f"preds.shape: {preds.shape}")
        logging.debug(f"trlabels.shape: {trlabels.shape}")
        
        self.model.load_state_dict(th.load(self.model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logging.info("Model loaded")
        all_head_idxs = self.ontology_classes_idxs.to(self.device)
        all_tail_idxs = self.ontology_classes_idxs.to(self.device)
        eval_rel_idx = None

        testing_dataloader = self.create_subsumption_dataloader(self.test_tuples_path, batch_size=self.test_batch_size)
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in tqdm(testing_dataloader, desc="Getting labels"):
                head_idxs = head_idxs.to(self.device)
                
                for i, head_graph_id in enumerate(head_idxs):
                    head_ont_id = th.where(self.ontology_classes_idxs == head_graph_id)[0]
                    rel = rel_idxs[i]
                    if self.graph_type == 'rdf':
                        graph_rel_name = self.id_to_class[rel.item()]    
                    else:
                        graph_rel_name = self.id_to_relation[rel.item()]
                    rel_id = self.eval_relations[graph_rel_name]
                    tail_graph_id = tail_idxs[i]
                    tail_ont_id = th.where(self.ontology_classes_idxs == tail_graph_id)[0]
                    trlabels[rel_id][head_ont_id][tail_ont_id] = 10000

        with th.no_grad():
            for eval_rel_idx in rel_to_eval_id.values():
                for (head_idxs,) in tqdm(self.prediction_dataloader(), desc=f"Getting predictions for relation {self.ontology_relations[eval_rel_idx]}"):
                    aux = head_idxs.to(self.device)

                    num_head_idxs = len(head_idxs)
                    head_idxs = head_idxs.to(self.device)
                    head_idxs = head_idxs.repeat(num_testing_tails,1).T
                    #assert (head_idxs[0,:] == aux[0]).all(), f"{head_idxs[0,:]}, {aux[0]}"

                    head_idxs = head_idxs.reshape(-1)
                    #assert (head_idxs[:num_testing_tails] == aux[0]).all(), f"{head_idxs[:num_testing_tails]}, {aux[0]}"
                    rel_idx = self.ontology_relations_idxs[eval_rel_idx]
                    rel_idxs = rel_idx * th.ones_like(head_idxs)

                    eval_tails = all_tail_idxs.repeat(num_head_idxs)
                    #assert (eval_tails[:num_testing_tails] == self.all_tail_idxs).all(), f"{eval_tails[:num_testing_tails]}, {self.all_tail_idxs}"

                    #assert head_idxs.shape == rel_idxs.shape == eval_tails.shape, f"{head_idxs.shape, rel_idxs.shape, eval_tails.shape}"

                    eval_tails = eval_tails.to(self.device)

                    if existential_axioms:
                        if self.graph_type == "rdf":
                            logits = self.rdf_existential_forward(head_idxs, rel_idxs, eval_tails, both_quantifiers)
                        else:
                            logits = self.existential_forward(head_idxs, rel_idxs, eval_tails, both_quantifiers)
                                
                    else:
                        logits = self.normal_forward(head_idxs, rel_idxs, eval_tails)
                                                
                    for i, head in enumerate(aux):
                        head_ont_id = th.where(self.ontology_classes_idxs == head)[0]
                        preds[eval_rel_idx][head_ont_id] = logits[i].cpu().numpy()

        assert np.min(preds) >=0, f"Min value of preds is {np.min(preds)}"
        
        return preds, trlabels

        #pkl.dump(self._predictions, open(self.pred_file, "wb"))
        #pkl.dump(self._labels, open(self.label_file, "wb"))
            

    def compute_ranking_metrics(self, predictions, training_labels):
        print(f"Loading best model from {self.model_path}")
        self.model.load_state_dict(th.load(self.model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        mean_rank, filtered_mean_rank = 0, 0
        mrr, filtered_mrr = 0, 0
        hits_at_1, fhits_at_1 = 0, 0
        hits_at_10, fhits_at_10 = 0, 0
        hits_at_100, fhits_at_100 = 0, 0
        ranks, filtered_ranks = dict(), dict()

        testing_dataloader = self.create_subsumption_dataloader(self.test_tuples_path, batch_size=self.test_batch_size)
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in tqdm(testing_dataloader, desc="Computing metrics..."):
                for i, graph_head in enumerate(head_idxs):

                    head = th.where(self.ontology_classes_idxs == graph_head)[0]
                    
                    graph_tail = tail_idxs[i]
                    tail = th.where(self.ontology_classes_idxs == graph_tail)[0]

                    rel = rel_idxs[i]
                    eval_rel = th.where(self.ontology_relations_idxs == rel)[0]
                    preds = predictions[eval_rel][head]

                    trlabels = training_labels[eval_rel][head]
                    trlabels[tail] = 1
                    filtered_preds = preds * trlabels
                                                            
                    preds = th.from_numpy(preds).to(self.device)
                    filtered_preds = th.from_numpy(filtered_preds).to(self.device)

                    orderings = th.argsort(preds, descending=False)
                    filtered_orderings = th.argsort(filtered_preds, descending=False)
                    
                    rank = th.where(orderings == tail)[0].item()
                    filtered_rank = th.where(filtered_orderings == tail)[0].item()
                    
                    mean_rank += rank
                    filtered_mean_rank += filtered_rank
                    
                    mrr += 1/(rank+1)
                    filtered_mrr += 1/(filtered_rank+1)
                    
                    if rank == 0:
                        hits_at_1 += 1
                    if rank < 10:
                        hits_at_10 += 1
                    if rank < 100:
                        hits_at_100 += 1

                    if rank not in ranks:
                        ranks[rank] = 0
                    ranks[rank] += 1

                    if filtered_rank == 0:
                        fhits_at_1 += 1
                    if filtered_rank < 10:
                        fhits_at_10 += 1
                    if filtered_rank < 100:
                        fhits_at_100 += 1

                    if filtered_rank not in filtered_ranks:
                        filtered_ranks[filtered_rank] = 0
                    filtered_ranks[filtered_rank] += 1

            mean_rank /= testing_dataloader.dataset_len
            mrr /= testing_dataloader.dataset_len
            hits_at_1 /= testing_dataloader.dataset_len
            hits_at_10 /= testing_dataloader.dataset_len
            hits_at_100 /= testing_dataloader.dataset_len
            auc = self.compute_rank_roc(ranks)

            filtered_mean_rank /= testing_dataloader.dataset_len
            filtered_mrr /= testing_dataloader.dataset_len
            fhits_at_1 /= testing_dataloader.dataset_len
            fhits_at_10 /= testing_dataloader.dataset_len
            fhits_at_100 /= testing_dataloader.dataset_len
            fauc = self.compute_rank_roc(filtered_ranks)

            raw_metrics = (mean_rank, mrr, hits_at_1, hits_at_10, hits_at_100, auc)
            filtered_metrics = (filtered_mean_rank, filtered_mrr, fhits_at_1, fhits_at_10, fhits_at_100, fauc)
        return raw_metrics, filtered_metrics


    def compute_rank_roc(self, ranks):
        n_tails = len(self.ontology_classes_idxs)
        auc_x = list(ranks.keys())
        auc_x.sort()
        auc_y = []
        tpr = 0
        sum_rank = sum(ranks.values())
        for x in auc_x:
            tpr += ranks[x]
            auc_y.append(tpr / sum_rank)
        auc_x.append(n_tails)
        auc_y.append(1)
        auc = np.trapz(auc_y, auc_x) / n_tails
        return auc


            
    def test(self, existential_axioms, both_quantifiers):
        """
        :param existential_axioms: If ``True``, the test is done over existential axioms. If ``False``, the test is done over subsumption axioms between named classes. 
        :type existential_axioms: bool
        :param both_quantifiers: If ``True``, the test is done by ranking over axioms with existential and universal axioms. If ``False``, the test only considers existential axioms.
        :type both_quantifiers: bool
        """
        print("\nTesting")
        print(f"\t\tBoth quantifiers: {both_quantifiers}")
        print(f"\t\tExistential axioms: {existential_axioms}")

        print("Computing predictions...")
        preds, trlabels = self.get_preds_and_labels(existential_axioms, both_quantifiers)
        print("Computing metrics...")
        raw_metrics, filtered_metrics = self.compute_ranking_metrics(preds, trlabels)
        return raw_metrics, filtered_metrics
        
                

                
                
