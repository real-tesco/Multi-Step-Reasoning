import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np


class MSMARCO(Dataset):
    # Triples either triples or pairs, refactor
    def __init__(self, pid2docid, examples, example_ids, train_time=True, dev_time=False):
        self.train = train_time
        self.dev = dev_time
        self.pid2docid = pid2docid

        if examples is not None:
            self._examples = examples
            self._example_ids = example_ids
            self._number_of_examples = len(examples)

    def __len__(self):
        if self.train:
            return self._number_of_examples
        elif self.dev:
            return self._number_of_examples
        else:
            return 0

    def __getitem__(self, idx):
        return self.vectorize(self, self._examples[idx])

    def vectorize(self, curr_example):
        if self.train:
            query = torch.FloatTensor(curr_example[0])
            positive = torch.FloatTensor(curr_example[1])
            negative = torch.FloatTensor(curr_example[2])

            return query, positive, negative  # qid, pid, nid, query, positive, negative
        elif self.dev:
            qid = curr_example[0]
            docid = curr_example[1]
            label = curr_example[2]
            query_embedding = curr_example[3]
            document_embedding = curr_example[4]
            rst = {'query_id': qid, 'doc_id': docid, 'label': label, 'query_embedding': query_embedding,
                   'document_embedding': document_embedding
            }
            return rst

    def get_docid(self, pid):
        return self.pid2docid[pid]

    def get_queries(self):
        return self.queries

    def get_query(self, qid):
        return self.queries[qid]
