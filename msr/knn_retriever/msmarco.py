import torch
import torch.utils.data
from torch.utils.data import Dataset


#TODO: look here

class MSMARCO(Dataset):
    def __init__(self, passages, pids, queries, qids, pid2docid):
        super(MSMARCO).__init__()
        self.number_of_passages = len(passages)
        self.number_of_queries = len(queries)
        self.queries = {}
        self.passages = {}
        self.pid2docid = pid2docid

        i = 0
        for passage in passages:
            self.passages[pids[i]] = passage
            i += 1
        i = 0
        for query in queries:
            self.queries[qids[i]] = query
            i += 1

    def __len__(self):
        return self.number_of_passages

    def __getitem__(self, item):
        return self.passages[item]

    def get_docid(self, pid):
        return self.pid2docid[pid]

    def get_queries(self):
        return self.queries

    def get_query(self, qid):
        return self.queries[qid]
