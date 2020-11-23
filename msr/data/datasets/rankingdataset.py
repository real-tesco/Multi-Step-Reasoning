from typing import List, Tuple, Dict, Any
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import logging

logger = logging.getLogger()


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

        # Load documents and convert to tensors
        tmp_docids = []
        tmp_docids.extend(np.load(x) for x in doc_ids_files)
        tmp_docids = np.concatenate(tmp_docids, axis=0)

        tmp_docs = []
        tmp_docs.extend(torch.tensor(np.load(x)) for x in doc_embedding_files)
        tmp_docs = torch.cat(tmp_docs, dim=0)

        self._docs = {idx: embed for idx, embed in zip(tmp_docids, tmp_docs)}

        tmp_query_ids = []
        tmp_query_ids.extend(np.load(x) for x in query_ids_files)
        tmp_query_ids = np.concatenate(tmp_query_ids, axis=0)
        tmp_queries = []
        tmp_queries.extend(torch.tensor(np.load(x)) for x in query_embedding_files)
        tmp_queries = torch.cat(tmp_queries, dim=0)

        self._queries = {idx: embed for idx, embed in zip(tmp_query_ids, tmp_queries)}

        #del tmp_queries, tmp_query_ids, tmp_docs, tmp_docids

        logger.info(f"len of docs: {len(self._docs)} | len of queries: {len(self._queries)}")

        self._dataset = dataset

        if self._dataset.split('.')[-1] == 'tsv':
            if isinstance(self._dataset, str):
                with open(self._dataset, 'r') as f:
                    self._examples = []
                    for i, line in enumerate(f):
                        line = line.strip().split()
                        self._examples.append(line)
        elif self._dataset.split('.')[-1] == 'jsonl':
            if isinstance(self._dataset, str):
                with open(self._dataset, 'r') as f:
                    self._examples = []
                    for i, line in enumerate(f):
                        line = json.loads(line)
                        self._examples.append(line)
        else:
            logger.info("unknown dataset name..")
        self._count = len(self._examples)
        logger.info(f"len of examples: {self._count}")

    def __getitem__(self, idx):
        example = self._examples[idx]
        if self._mode == 'train':
            return {'query': self._queries[example[0]],
                    'positive_doc': self._docs[example[1]],
                    'negative_doc': self._docs[example[2]]}
        elif self._mode == 'dev':
            query_id = example['query_id']
            doc_id = example['doc_id']
            retrieval_score = example['retrieval_score']
            label = example['label']
            return {'query_id': query_id, 'doc_id': doc_id, 'label': label, 'retrieval_score': retrieval_score,
                    'query': self._queries[query_id], 'doc': self._docs[doc_id]}

    def collate(self, batch):
        if self._mode == 'train':
            #logger.info(f'COLLATE batch: {batch} | query: {[item["query"][:10] for item in batch]}')
            queries = torch.stack([item['query'] for item in batch])
            positive_docs = torch.stack([item['positive_doc'] for item in batch])
            negative_docs = torch.stack([item['negative_doc'] for item in batch])
            return {'query': queries, 'positive_doc': positive_docs, 'negative_doc': negative_docs}
        elif self._mode == 'dev':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id']for item in batch]
            retrieval_score = [item['retrieval_score'] for item in batch]
            labels = [item['label'] for item in batch]
            queries = torch.tensor([item['query'] for item in batch])
            docs = torch.tensor([item['doc'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'label': labels, 'retrieval_score': retrieval_score,
                    'doc': docs, 'query': queries}

    def __len__(self):
        return self._count
