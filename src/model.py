import os
import pandas as pd
import numpy as np
from src.utils import FastTensorDataLoader

import torch as th
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn

#from src.OpenKE.openke.module.model import TransE
from pykeen.models import TransE, TransR, TransD, BoxE
from pykeen.triples import TriplesFactory

from tqdm import trange, tqdm

from mowl.owlapi.defaults import BOT, TOP

prefix = {
    "go": "go",
    "hpo": "hp",
    
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
        self.kg_module =  TransD(triples_factory=self.triples_factory,
                                 embedding_dim=self.embedding_dim,
                                 #scoring_fct_norm=2, #self.p_norm, # Trans[E,R]
                                 random_seed = self.random_seed)
                        
    def forward(self, data, mode="kg"):
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

        self._triples_factory = None

        self.model = ProjectionModule(triples_factory=self.triples_factory,
                            embedding_dim=self.emb_dim,
                            #scoring_fct_norm=self.p_norm, # Trans[E,R]
                            random_seed = self.seed
                            )
        #self.model = TransE(len(self.class_to_id),
        #                    len(self.relation_to_id),
        #                    dim = self.emb_dim,
        #                    p_norm = self.p_norm,
        #                    norm_flag = True,
        #                    margin = None,
        #                    epsilon = None)

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
        return self._class_to_id

    @property
    def relation_to_id(self):
        if self._relation_to_id is not None:
            return self._relation_to_id

        graph_rels = list(self.graph["relation"].unique())
        graph_rels.sort()
        self._relation_to_id = {r: i for i, r in enumerate(graph_rels)}
        return self._relation_to_id
        

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
            
    # @property
    # def inferred_ancestors(self):
    #     if self._inferred_ancestors is not None:
    #         return self._inferred_ancestors
    #     print("Loading inferred ancestors from disk...")
    #     path = os.path.join(self.root, "inferred_ancestors.txt")
    #     assert os.path.exists(path), f"Inferred ancestors file {path} does not exist"

    #     self._inferred_ancestors = dict()
    #     with open(path, "r") as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             line = line.rstrip("\n").split(",")
    #             entity = line[0]
    #             ancestors = line[1:]
    #             self._inferred_ancestors[entity] = ancestors
    #     print("Done")
    #     return self._inferred_ancestors
            
    # @property
    # def inferred_ancestors_tensor(self):
    #     if self._inferred_ancestors_tensor is not None:
    #         return self._inferred_ancestors_tensor

    #     self._inferred_ancestors_tensor = dict()

    #     for entity, ancestors in self.inferred_ancestors.items():
    #         idxs = [self.class_to_id[x] for x in ancestors]
    #         idxs = [th.nonzero(self.ontology_classes == x, as_tuple=False).item() for x in idxs]

    #         tensor = th.ones(len(self.ontology_classes))
    #         tensor[idxs] = 100000
    #         self._inferred_ancestors_tensor[self.class_to_id[entity]] = tensor

    #     return self._inferred_ancestors_tensor
        
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
            rels = [self.relation_to_id[r] for r in tuples["relation"]]
            rels = th.tensor(rels, dtype=th.long)
        
        dataloader = FastTensorDataLoader(heads, rels, tails, batch_size=batch_size, shuffle=True)
        return dataloader

    def create_existential_dataloader(self, tuples_path, batch_size):
        tuples = pd.read_csv(tuples_path, sep=",", header=None)
 
        tuples.columns = ["head", "relation", "tail"]
                        
        heads = [self.class_to_id[h] for h in tuples["head"]]
        rels = [self.relation_to_id[r] for r in tuples["relation"]]
        tails = [self.class_to_id[t] for t in tuples["tail"]]
        
        heads = th.tensor(heads, dtype=th.long)
        rels = th.tensor(rels, dtype=th.long)
        tails = th.tensor(tails, dtype=th.long)

        dataloader = FastTensorDataLoader(heads, rels, tails, batch_size=batch_size, shuffle=True)
        return dataloader

                                                                                                            
    def train(self):

        optimizer = optim.Adam(self.model.parameters(), lr=0.000001, weight_decay = self.weight_decay)
        min_lr = self.lr/100  #0.0001 #0.000001
        max_lr = self.lr #0.01 #0.0001
        scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up = 30, cycle_momentum = False)
        criterion = nn.MarginRankingLoss(margin=self.margin)
        criterion_bpr = nn.LogSigmoid()
        criterion_bce = nn.BCEWithLogitsLoss()
        
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
                pos_logits = self.model.forward(data, mode="kg")

                neg_logits = 0
                for i in range(self.num_negs):
                    neg_tail = th.randint(0, len(self.class_to_id), (len(head),), device=self.device)
                    data = (head, rel, neg_tail)
                    neg_logits += self.model.forward(data, mode="kg")
                neg_logits /= self.num_negs
                
                batch_loss= -criterion_bpr(pos_logits - neg_logits + self.margin).mean()

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
                

    def test(self, mode="subsumption"):

        if not mode in ["subsumption", "existential"]:
            raise ValueError("Mode must be either subsumption or existential")
        
        print(f"Loading best model from {self.model_path}")
        print("\nTesting")
        self.model.load_state_dict(th.load(self.model_path))
        self.model = self.model.to(self.device)

        if mode == "subsumption":
            test_subsumption_dl = self.create_subsumption_dataloader(self.test_tuples_path, batch_size=self.test_batch_size)
        elif mode == "existential":
            test_subsumption_dl = self.create_existential_dataloader(self.test_tuples_path, batch_size=self.test_batch_size)
            
        self.model.eval()
                                                
        num_ontology_classes = len(self.ontology_classes_idxs)

        print("Number of classes:", num_ontology_classes)

        hits_at_1 = 0
        hits_at_10 = 0
        hits_at_100 = 0
        
        mean_rank = 0
        mrr = 0

        all_classes = th.tensor(list(self.ontology_classes_idxs), dtype=th.long, device=self.device)
        with th.no_grad():
            for heads, rels, tails in tqdm(test_subsumption_dl):
                                                    
                aux = heads.to(self.device)
                num_heads = len(heads)
                
                heads = heads.to(self.device)
                heads = heads.repeat(num_ontology_classes,1).T

                #assert (heads[0,:] == aux[0]).all(), f"{heads[0,:]}, {aux[0]}"
                heads = heads.reshape(-1)
                #assert (heads[:num_ontology_classes] == aux[0]).all(), f"{heads[:num_ontology_classes]}, {aux[0]}"

                rels = rels.to(self.device)
                rels = rels.repeat(num_ontology_classes,1).T
                rels = rels.reshape(-1)
                                                
                eval_tails = self.ontology_classes_idxs.repeat(num_heads)
                #assert (eval_tails[:num_ontology_classes] == all_classes).all(), f"{eval_tails[:num_ontology_classes]}, {self.ontology_classes}"
                
                #assert heads.shape == eval_tails.shape == rels.shape, f"{heads.shape} {eval_tails.shape} {rels.shape}"
                 
                data = (heads, rels, eval_tails)
                logits = self.model.forward(data, mode="kg")
                logits = logits.reshape(num_heads, num_ontology_classes)

                orderings = th.argsort(logits, dim=1, descending=True)

                all_classes_repeated = all_classes.repeat(len(tails),1)
                tail_ids = th.nonzero(all_classes_repeated == tails.to(self.device).unsqueeze(1), as_tuple=False)[:,1]
                                                                            
                ranks = th.nonzero(orderings == tail_ids.unsqueeze(1), as_tuple=False)[:,1]

                for rank in ranks:
                    rank = rank.item()
                    mean_rank += rank
                    mrr += (1/(rank+1))
                    if rank == 0:
                        hits_at_1 += 1
                    if rank < 10:
                        hits_at_10 += 1
                    if rank < 100:
                        hits_at_100 += 1
                                            
                        
        mean_rank /= test_subsumption_dl.dataset_len
        mrr /= test_subsumption_dl.dataset_len
        hits_at_1 /= test_subsumption_dl.dataset_len
        hits_at_10 /= test_subsumption_dl.dataset_len
        hits_at_100 /= test_subsumption_dl.dataset_len
        return mean_rank, mrr, hits_at_1, hits_at_10, hits_at_100


    def test_filtered(self, mode="subsumption"):

        if not mode in ["subsumption", "existential"]:
            raise ValueError("Mode must be either subsumption or existential")
        
        print(f"Loading best model from {self.model_path}")
        print("\nTesting")
        self.model.load_state_dict(th.load(self.model_path))
        self.model = self.model.to(self.device)

        if mode == "subsumption":
            test_subsumption_dl = self.create_subsumption_dataloader(self.test_tuples_path, batch_size=self.test_batch_size)
        elif mode == "existential":
            test_subsumption_dl = self.create_existential_dataloader(self.test_tuples_path, batch_size=self.test_batch_size)
            
        self.model.eval()
                                                
        num_ontology_classes = len(self.ontology_classes_idxs)

        print("Number of classes:", num_ontology_classes)

        hits_at_1 = 0
        hits_at_10 = 0
        hits_at_100 = 0
        
        mean_rank = 0
        mrr = 0

        test_set = dict()
        all_classes = th.tensor(list(self.ontology_classes_idxs), dtype=th.long, device=self.device)
        print("Getting filtering data...")
        for heads, rels, tails in tqdm(test_subsumption_dl):
            for i, head in enumerate(heads):
                h = th.where(all_classes == head)[0].item()
                r = rels[i].item()
                t = th.where(all_classes == tails[i])[0].item()
                
                if (h,r) not in test_set:
                    test_set[(h,r)] = set()
                test_set[(h,r)].add(t)
        
        
        print("Testing...")
        with th.no_grad():
            for heads, rels, tails in tqdm(test_subsumption_dl):
                                                    
                aux = heads.to(self.device)
                num_heads = len(heads)
                
                heads = heads.to(self.device)
                heads = heads.repeat(num_ontology_classes,1).T

                #assert (heads[0,:] == aux[0]).all(), f"{heads[0,:]}, {aux[0]}"
                heads = heads.reshape(-1)
                #assert (heads[:num_ontology_classes] == aux[0]).all(), f"{heads[:num_ontology_classes]}, {aux[0]}"

                rels = rels.to(self.device)
                rels_rep = rels.repeat(num_ontology_classes,1).T
                rels_rep = rels_rep.reshape(-1)
                                                
                eval_tails = self.ontology_classes_idxs.repeat(num_heads)
                #assert (eval_tails[:num_ontology_classes] == all_classes).all(), f"{eval_tails[:num_ontology_classes]}, {self.ontology_classes}"
                
                #assert heads.shape == eval_tails.shape == rels_rep.shape, f"{heads.shape} {eval_tails.shape} {rels_rep.shape}"
                 
                data = (heads, rels_rep, eval_tails)
                logits = self.model.forward(data, mode="kg")
                logits = logits.reshape(num_heads, num_ontology_classes)
                tails = tails.to(self.device)
                for i, head in enumerate(aux):
                    head_id = th.where(all_classes==head)[0].item()
                    rel_id = rels[i].item()
                    for cand_tail in list(test_set[(head_id, rel_id)]):
                        tail_id = th.where(all_classes==tails[i])[0].item()
                        if cand_tail == tail_id:
                            continue
                        logits[i, cand_tail] = -1e9

                orderings = th.argsort(logits, dim=1, descending=True)

                all_classes_repeated = all_classes.repeat(len(tails),1)
                tail_ids = th.nonzero(all_classes_repeated == tails.to(self.device).unsqueeze(1), as_tuple=False)[:,1]
                                                                            
                ranks = th.nonzero(orderings == tail_ids.unsqueeze(1), as_tuple=False)[:,1]

                for rank in ranks:
                    rank = rank.item()
                    mean_rank += rank
                    mrr += (1/(rank+1))
                    if rank == 0:
                        hits_at_1 += 1
                    if rank < 10:
                        hits_at_10 += 1
                    if rank < 100:
                        hits_at_100 += 1
                                            
                        
        mean_rank /= test_subsumption_dl.dataset_len
        mrr /= test_subsumption_dl.dataset_len
        hits_at_1 /= test_subsumption_dl.dataset_len
        hits_at_10 /= test_subsumption_dl.dataset_len
        hits_at_100 /= test_subsumption_dl.dataset_len
        return mean_rank, mrr, hits_at_1, hits_at_10, hits_at_100



        
                

                
                
