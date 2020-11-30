from typing import List, Tuple, Dict, Any

import json

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class BertDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: AutoTokenizer,
        mode: str,
        query_max_len: int = 32,
        doc_max_len: int = 256,
        max_input: int = 1280000,
            ) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._mode = mode
        self._query_max_len = query_max_len
        self._doc_max_len = doc_max_len
        self._seq_max_len = doc_max_len
        self._max_input = max_input
        if self._seq_max_len > 512:
            raise ValueError('doc_max_len > 512.')

        if isinstance(self._dataset, str):
            self._id = False
            with open(self._dataset, 'r') as f:
                self._examples = []
                for i, line in enumerate(f):
                    if i >= self._max_input:
                        break
                    line = json.loads(line)
                    self._examples.append(line)
        elif isinstance(self._dataset, dict):
            self._id = True
            self._queries = {}
            with open(self._dataset['queries'], 'r') as f:
                for line in f:
                    if self._dataset['queries'].split('.')[-1] == 'json' or self._dataset['queries'].split('.')[-1] == 'jsonl':
                        line = json.loads(line)
                    else:
                        query_id, query = line.strip('\n').split('\t')
                        line = {'query_id': query_id, 'query': query}
                    self._queries[line['query_id']] = line['query']
            self._docs = {}
            with open(self._dataset['docs'], 'r') as f:
                for line in f:
                    if self._dataset['docs'].split('.')[-1] == 'json' or self._dataset['docs'].split('.')[-1] == 'jsonl':
                        line = json.loads(line)
                    else:
                        doc_id, _, _, doc = line.strip('\n').split('\t')
                        line = {'doc_id': doc_id, 'doc': doc}
                    self._docs[line['doc_id']] = line['doc']
            if self._mode == 'dev':
                qrels = {}
                with open(self._dataset['qrels'], 'r') as f:
                    for line in f:
                        line = line.strip().split()
                        if line[0] not in qrels:
                            qrels[line[0]] = {}
                        qrels[line[0]][line[2]] = int(line[3])
            with open(self._dataset['trec'], 'r') as f:
                self._examples = []
                for i, line in enumerate(f):
                    if i >= self._max_input:
                        break
                    line = line.strip().split()
                    if self._mode == 'dev':
                        if line[0] not in qrels or line[2] not in qrels[line[0]]:
                            label = 0
                        else:
                            label = qrels[line[0]][line[2]]
                    if self._mode == 'train':
                        self._examples.append({'query_id': line[0], 'doc_id': line[1], 'label': int(line[2])})
                    elif self._mode == 'dev':
                        self._examples.append({'label': label, 'query_id': line[0], 'doc_id': line[2], 'retrieval_score': float(line[4])})
                    elif self._mode == 'test':
                        self._examples.append({'query_id': line[0], 'doc_id': line[2], 'retrieval_score': float(line[4])})
                    else:
                        raise ValueError('Mode must be `train`, `dev` or `test`.')
        else:
            raise ValueError('Dataset must be `str` or `dict`.')
        self._count = len(self._examples)

    def collate(self, batch: Dict[str, Any]):
        if self._mode == 'train':

            q_input_ids = torch.tensor([item['q_input_ids'] for item in batch])
            q_segment_ids = torch.tensor([item['q_segment_ids'] for item in batch])
            q_input_mask = torch.tensor([item['q_input_mask'] for item in batch])
            d_input_ids = torch.tensor([item['d_input_ids'] for item in batch])
            d_segment_ids = torch.tensor([item['d_segment_ids'] for item in batch])
            d_input_mask = torch.tensor([item['d_input_mask'] for item in batch])
            label = torch.tensor([item['label'] for item in batch])
            return {'q_input_ids': q_input_ids, 'q_segment_ids': q_segment_ids, 'q_input_mask': q_input_mask,
                    'd_input_ids': d_input_ids, 'd_segment_ids': d_segment_ids, 'd_input_mask': d_input_mask, 'label': label}

        elif self._mode == 'dev':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            label = [item['label'] for item in batch]
            retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
            q_input_ids = torch.tensor([item['q_input_ids'] for item in batch])
            q_segment_ids = torch.tensor([item['q_segment_ids'] for item in batch])
            q_input_mask = torch.tensor([item['q_input_mask'] for item in batch])
            d_input_ids = torch.tensor([item['d_input_ids'] for item in batch])
            d_segment_ids = torch.tensor([item['d_segment_ids'] for item in batch])
            d_input_mask = torch.tensor([item['d_input_mask'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'label': label, 'retrieval_score': retrieval_score,
                    'q_input_ids': q_input_ids, 'q_segment_ids': q_segment_ids, 'q_input_mask': q_input_mask,
                    'd_input_ids': d_input_ids, 'd_segment_ids': d_segment_ids, 'd_input_mask': d_input_mask}
        elif self._mode == 'test':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            segment_ids = torch.tensor([item['segment_ids'] for item in batch])
            input_mask = torch.tensor([item['input_mask'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'retrieval_score': retrieval_score,
                    'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask}
        elif self._mode == 'embed':
            doc_id = [item['doc_id'] for item in batch]
            input_ids = torch.tensor([item['d_input_ids'] for item in batch])
            segment_ids = torch.tensor([item['d_segment_ids'] for item in batch])
            input_mask = torch.tensor([item['d_input_mask'] for item in batch])
            return {'doc_id': doc_id, 'd_input_ids': input_ids, 'd_segment_ids': segment_ids,
                    'd_input_mask': input_mask}
        elif self._mode == 'inference':
            query_id = [item['query_id'] for item in batch]
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            segment_ids = torch.tensor([item['segment_ids'] for item in batch])
            input_mask = torch.tensor([item['input_mask'] for item in batch])
            return {'query_id': query_id, 'q_input_ids': input_ids, 'q_segment_ids': segment_ids,
                    'q_input_mask': input_mask}
        else:
            raise ValueError('Mode must be `train`, `dev`, `test` or `embed`.')

    def pack_bert_features(self, query_tokens: List[str], doc_tokens: List[str], two_tower=False):
        if not two_tower:
            input_tokens = [self._tokenizer.cls_token] + query_tokens + [self._tokenizer.sep_token] + doc_tokens + [self._tokenizer.sep_token]
            input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_tokens) + 1)
            input_mask = [1] * len(input_tokens)

            padding_len = self._seq_max_len - len(input_ids)
            input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
            input_mask = input_mask + [0] * padding_len
            segment_ids = segment_ids + [0] * padding_len

            assert len(input_ids) == self._seq_max_len
            assert len(input_mask) == self._seq_max_len
            assert len(segment_ids) == self._seq_max_len

            return input_ids, input_mask, segment_ids
        else:
            tokenized = []
            for idx, tokens in enumerate([query_tokens, doc_tokens]):
                input_tokens = [self._tokenizer.cls_token] + tokens + [self._tokenizer.sep_token]
                input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
                segment_ids = [1] * len(input_ids)
                input_mask = [1] * len(input_ids)
                if idx == 0:
                    padding_len = self._query_max_len - len(input_ids)
                else:
                    padding_len = self._seq_max_len - len(input_ids)
                input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
                input_mask = input_mask + [0] * padding_len
                segment_ids = segment_ids + [0] * padding_len
                if idx == 0:
                    assert len(input_ids) == self._query_max_len
                    assert len(input_mask) == self._query_max_len
                    assert len(segment_ids) == self._query_max_len
                else:
                    assert len(input_ids) == self._seq_max_len
                    assert len(input_mask) == self._seq_max_len
                    assert len(segment_ids) == self._seq_max_len
                tokenized.append((input_ids, segment_ids, input_mask))
            return tokenized

    def pack_bert_features_doc_only(self, doc_tokens: List[str]):
        input_tokens = [self._tokenizer.cls_token] + doc_tokens + [self._tokenizer.sep_token]
        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        segment_ids = [1] * len(input_ids)
        input_mask = [1] * len(input_ids)

        padding_len = self._seq_max_len - len(input_ids)

        input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
        input_mask = input_mask + [0] * padding_len
        segment_ids = segment_ids + [0] * padding_len

        assert len(input_ids) == self._seq_max_len
        assert len(input_mask) == self._seq_max_len
        assert len(segment_ids) == self._seq_max_len
        return input_ids, segment_ids, input_mask

    def pack_bert_features_q_only(self, q_tokens: List[str]):
        input_tokens = [self._tokenizer.cls_token] + q_tokens + [self._tokenizer.sep_token]
        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        segment_ids = [1] * len(input_ids)
        input_mask = [1] * len(input_ids)

        padding_len = self._query_max_len - len(input_ids)

        input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
        input_mask = input_mask + [0] * padding_len
        segment_ids = segment_ids + [0] * padding_len

        assert len(input_ids) == self._query_max_len
        assert len(input_mask) == self._query_max_len
        assert len(segment_ids) == self._query_max_len
        return input_ids, segment_ids, input_mask

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        if self._id:
            example['query'] = self._queries[example['query_id']]
            example['doc'] = self._docs[example['doc_id']]

        if self._mode == 'train':

            query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len-2]
            doc_tokens = self._tokenizer.tokenize(example['doc'])[:self._seq_max_len-2]
            tokenized = self.pack_bert_features(query_tokens, doc_tokens, two_tower=True)
            return {'q_input_ids': tokenized[0][0], 'q_segment_ids': tokenized[0][1], 'q_input_mask': tokenized[0][2],
                    'd_input_ids': tokenized[1][0], 'd_segment_ids': tokenized[1][1], 'd_input_mask': tokenized[1][2],
                    'label': example['label']}

        elif self._mode == 'dev':
            query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len-2]
            doc_tokens = self._tokenizer.tokenize(example['doc'])[:self._seq_max_len-2]

            tokenized = self.pack_bert_features(query_tokens, doc_tokens, two_tower=True)
            return {'query_id': example['query_id'], 'doc_id': example['doc_id'], 'label': example['label'], 'retrieval_score': example['retrieval_score'],
                    'q_input_ids': tokenized[0][0], 'q_segment_ids': tokenized[0][1], 'q_input_mask': tokenized[0][2],
                    'd_input_ids': tokenized[1][0], 'd_segment_ids': tokenized[1][1], 'd_input_mask': tokenized[1][2]}
        elif self._mode == 'test':
            query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
            doc_tokens = self._tokenizer.tokenize(example['doc'])[:self._seq_max_len-len(query_tokens)-3]

            input_ids, input_mask, segment_ids = self.pack_bert_features(query_tokens, doc_tokens)
            return {'query_id': example['query_id'], 'doc_id': example['doc_id'], 'retrieval_score': example['retrieval_score'],
                    'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids}
        elif self._mode == 'embed':
            doc_tokens = self._tokenizer.tokenize(example['doc'])[:self._seq_max_len-2]
            tokenized = self.pack_bert_features_doc_only(doc_tokens)
            return {'doc_id': example['doc_id'], 'd_input_ids': tokenized[0], 'd_segment_ids': tokenized[1],
                    'd_input_mask': tokenized[2]}
        elif self._mode == 'inference':
            query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len-2]
            input_ids, input_mask, segment_ids = self.pack_bert_features_q_only(query_tokens)
            return {'query_id': example['query_id'],
                    'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids}
        else:
            raise ValueError('Mode must be `train`, `dev`, `test`, `embed`, or `inference`.')

    def __len__(self) -> int:
        return self._count
