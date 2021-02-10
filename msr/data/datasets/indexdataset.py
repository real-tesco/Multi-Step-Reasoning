from typing import Dict, Any
import numpy as np
from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        indices: str
            ) -> None:
        self._dataset = dataset
        self._indices = indices

        self._examples = np.load(self._dataset)
        self._indices = np.load(self._indices)
        self._num_docs = len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rst = {'id': self._indices[idx],
               'doc': self._examples[idx]}
        return rst

    def collate(self, batch: Dict[str, Any]):
        doc_ids = np.asarray([item['id'] for item in batch]).astype(np.float32)
        docs = np.asarray([item['doc'] for item in batch]).astype(np.float32)
        return {'id': doc_ids, 'doc': docs}
