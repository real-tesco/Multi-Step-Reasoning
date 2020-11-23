from typing import List, Tuple, Dict, Any
import json
import torch
from torch.utils.data import Dataset
import numpy as np


class RankingDataset(Dataset):
    def __init__(
        self,
        doc_embedding_files: List,
        doc_ids_files: List,
        query_embedding_files: List,
        query_ids_files: List,
        dataset: str,
        mode: str = 'train'
            ) -> None:
        self._mode = mode
        self._doc_ids = []
        self._docs = {}
        self._queries = {}
        self._query_ids = []

        # Load documents and convert to tensors
        tmp_docids = []
        tmp_docids.extend(np.load(x) for x in doc_ids_files)
        tmp_docids = np.concatenate(tmp_docids, axis=0)
        print(tmp_docids.shape)

        tmp_docs = []
        tmp_docs.extend(torch.tensor(np.load(x)) for x in doc_embedding_files)
        tmp_docs = torch.cat(tmp_docs, dim=0)
        print(tmp_docs.shape)
        #self._docs[self._doc_ids[i]] = tmp_docs[i] for i in range(0, len(tmp_docs))
        self._docs = {idx: embed for idx, embed in zip(tmp_docids, tmp_docs)}

        print(self._docs.keys())
        print(len(self._docs))

        self._query_ids.extend(np.load(x) for x in query_ids_files)
        tmp_queries = []
        tmp_queries.extend(torch.tensor(np.load(x)) for x in query_embedding_files)
        tmp_queries = torch.cat(tmp_queries)
        print(tmp_queries.shape)


        self._dataset = dataset

        if isinstance(self._dataset, str):
            with open(self._dataset, 'r') as f:
                self._examples = []
                for i, line in enumerate(f):
                    line = line.strip().split()
                    self._examples.append(line)
        self._count = len(self._examples)

    def __getitem__(self, idx):
        example = self._examples[idx]
        if self._mode == 'train':
            return {'query': example[0], 'positive_doc': example[1], 'negative_doc': example[2]}
        elif self._mode == 'dev':
            return {}

    def collate(self, batch):
        if self._mode == 'train':
            queries = torch.tensor([item['query'] for item in batch])
            positive_docs = torch.tensor([item['positive_doc'] for item in batch])
            negative_docs = torch.tensor([item['negative_doc'] for item in batch])
            return {'query': queries, 'positive_doc': positive_docs, 'negative_doc': negative_docs}
        elif self._mode == 'dev':
            return {}

    def __len__(self):
        return self._count
