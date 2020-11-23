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
        self._docs = []
        self._queries = []
        self._query_ids = []

        # Load documents and convert to tensors
        print(doc_ids_files[:3])
        self._doc_ids.extend(np.load(x).tolist() for x in doc_ids_files)
        print(len(self._doc_ids))
        #print(self._doc_ids[:3])
        self._docs.extend(torch.tensor(np.load(x)) for x in doc_embedding_files)
        self._docs = torch.cat(self._docs, dim=0)
        print(self._docs.shape)

        self._query_ids.extend(np.load(x) for x in query_ids_files)
        self._queries.extend(torch.tensor(np.load(x)) for x in query_embedding_files)
        self._queries = torch.cat(self._queries)
        print(self._queries.shape)

        self._dataset = dataset

        if isinstance(self._dataset, str):
            with open(self._dataset, 'r') as f:
                self._examples = []
                for i, line in enumerate(f):
                    if i >= self._max_input:
                        break
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
