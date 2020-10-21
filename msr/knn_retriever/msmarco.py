import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np

#TODO: look here

class MSMARCO(Dataset):
    def __init__(self, pid2docid, triples, triple_ids, train_time=True):
        self.train = train_time
        #self.number_of_queries = len(queries)
        self.qrels = {}
        #self.queries = {}
        #self.passages = {}
        self.pid2docid = pid2docid
        self.triples = triples
        self.triple_ids = triple_ids
        self.number_of_examples = len(triples)

        #i = 0
        #for query in queries:
        #    self.queries[qids[i]] = query
        #    i += 1

    def __len__(self):
        if self.train:
            return self.number_of_examples
        else:
            return 0

    def __getitem__(self, idx):
        if self.train:
            return self.vectorize_(self.triples[idx], self.triple_ids[idx])
        else:
            return self.passages[idx]

    def vectorize_(self, curr_triple, triple_ids):

        assert len(curr_triple) == len(triple_ids) == 3
        qid = triple_ids[0]
        pid = triple_ids[1]
        nid = triple_ids[2]
        query = torch.FloatTensor(curr_triple[0])
        positive = torch.FloatTensor(curr_triple[1])
        negative = torch.FloatTensor(curr_triple[2])

        return qid, pid, nid, query, positive, negative

    def get_docid(self, pid):
        return self.pid2docid[pid]

    def get_queries(self):
        return self.queries

    def get_query(self, qid):
        return self.queries[qid]
