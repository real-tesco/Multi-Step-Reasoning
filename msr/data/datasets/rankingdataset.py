from typing import List, Tuple, Dict, Any
import json
import torch
from torch.utils.data import Dataset
import numpy as np


class RankingDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        mode: str = 'train'
            ) -> None:
        self._dataset = dataset
        self._mode = mode
        self._examples = np.load(dataset).astype(np.float32)
        self._examples2 = np.load(dataset).astype(np.float32)
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
