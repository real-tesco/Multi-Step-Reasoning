import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np


def vectorize_(curr_triple, triple_ids):
    assert len(curr_triple) == len(triple_ids) == 3
    qid = triple_ids[0]
    pid = triple_ids[1]
    nid = triple_ids[2]
    query = torch.FloatTensor(curr_triple[0])
    positive = torch.FloatTensor(curr_triple[1])
    negative = torch.FloatTensor(curr_triple[2])

    return qid, pid, nid, query, positive, negative


def vectorize_query_(curr_query, curr_qid):
    return curr_qid, torch.FloatTensor(curr_query)


class MSMARCO(Dataset):
    def __init__(self, pid2docid, triples, triple_ids, train_time=True, dev_time=False):
        self.train = train_time
        self.dev = dev_time
        self.pid2docid = pid2docid

        if triples is not None:
            self.triples = triples
            self.triple_ids = triple_ids
            self.number_of_examples = len(triples)

    def __len__(self):
        if self.train:
            return self.number_of_examples
        elif self.dev:
            return self.number_of_queries
        else:
            return 0

    def __getitem__(self, idx):
        if self.train:
            return vectorize_(self.triples[idx], self.triple_ids[idx])
        elif self.dev:
            return vectorize_query_(self.queries[idx], self.query_ids[idx])
        else:
            return None

    def get_docid(self, pid):
        return self.pid2docid[pid]

    def get_queries(self):
        return self.queries

    def get_query(self, qid):
        return self.queries[qid]
