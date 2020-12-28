from typing import Dict, Any

import json

import torch
from torch.utils.data import Dataset


class BM25Dataset(Dataset):
    def __init__(
        self,
        dataset: str,
            ) -> None:
        self._dataset = dataset

        if isinstance(self._dataset, str):
            with open(self._dataset, 'r') as f:
                self._examples = []
                for line in f:
                    line = json.loads(line)
                    self._examples.append(line)
        self._count = len(self._examples)

    def collate(self, batch: Dict[str, Any]):
        query_ids = torch.tensor([item['query_id'] for item in batch])
        queries = torch.tensor([item['query'] for item in batch])

        return {'query_id': query_ids, "query": queries}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        return {"query_id": example['query_id'], "query": example['query']}

    def __len__(self) -> int:
        return self._count
