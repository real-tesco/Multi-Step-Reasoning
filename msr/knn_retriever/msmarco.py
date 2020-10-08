import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np

#TODO: look here

class MSMARCO(Dataset):
    def __init__(self, passages, pids, queries, qids, pid2docid, qrels, triples, train_time=False):
        self.train = train_time
        self.number_of_passages = len(passages)
        self.number_of_queries = len(queries)
        self.qrels = {}
        self.queries = {}
        self.passages = {}
        self.pid2docid = pid2docid
        self.triples = triples
        self.number_of_examples = len(triples)

        i = 0
        for passage in passages:
            self.passages[pids[i]] = passage
            i += 1
        i = 0
        for query in queries:
            self.queries[qids[i]] = query
            i += 1

    def __len__(self):
        if self.train:
            return self.number_of_examples
        else:
            return self.number_of_passages

    def __getitem__(self, idx):
        if self.train:
            return self.vectorize_(self.triples[idx])
        else:
            return self.passages[idx]

    def vectorize_(self, curr_triple):

        qid = curr_triple['qid']
        pid = curr_triple['pos']
        nid = curr_triple['neg']
        query = torch.LongTensor(self.queries[qid])
        positive = torch.LongTensor(self.passages[pid])
        negative = torch.LongTensor(self.passages[nid])

        return qid, pid, nid, query, positive, negative


    def get_docid(self, pid):
        return self.pid2docid[pid]

    def get_queries(self):
        return self.queries

    def get_query(self, qid):
        return self.queries[qid]
