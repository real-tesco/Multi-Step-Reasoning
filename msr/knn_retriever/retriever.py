import logging

import json
import numpy as np
import hnswlib
import torch
from transformers import AutoTokenizer


logger = logging.getLogger()


class KnnIndex:
    def __init__(self, args, model):
        self._args = args
        self._seq_max_len = args.max_doc_len
        self._query_max_len = args.max_query_len
        self._index = hnswlib.Index(space=args.similarity, dim=args.dim_hidden)
        self._tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
        self._model = model
        self._docid2indexid = {}
        self._indexid2docid = {}
        self._remainP = args.remainP

        with open(args.index_mapping, 'r') as f:
            mapping = json.load(f)
        for key in mapping:
            if self._remainP:
                self._indexid2docid[mapping[key][0]] = key
                self._indexid2docid[mapping[key][1]] = key

                # docid2indexid has lists as values
                self._docid2indexid[key] = mapping[key]
            else:
                self._indexid2docid[mapping[key]] = key
                self._docid2indexid[key] = mapping[key]

    def knn_query_text(self, query_text, device, k=100):
        input_ids, segment_ids, input_mask = self.tokenize(query_text)
        return self.knn_query_inference(input_ids.to(device), segment_ids.to(device), input_mask.to(device), k=k)

    def knn_query_embedded(self, query_embedding, k=100):
        query = query_embedding.detach().numpy()
        labels, distances = self._index.knn_query(query, k=k)
        distances = distances.tolist()
        document_labels = [[self._indexid2docid[labels[j][i]] for i in range(len(labels[j]))] for j in
                           range(len(labels))]
        document_embeddings = torch.tensor(self._index.get_items(labels.flatten()))
        document_embeddings = document_embeddings.reshape(labels.shape[0], labels.shape[1], self._args.dim_hidden)
        return document_labels, document_embeddings, distances, query_embedding

    def knn_query_inference(self, q_input_ids, q_segment_ids, q_input_mask, k=100):
        query_embedding = self._model.calculate_embedding(q_input_ids, q_segment_ids, q_input_mask, doc=False)
        labels, distances = self._index.knn_query(query_embedding.detach().cpu().numpy(), k=k)
        distances = distances.tolist()
        document_labels = [[self._indexid2docid[labels[j][i]] for i in range(len(labels[j]))] for j in
                           range(len(labels))]
        document_embeddings = torch.tensor(self._index.get_items(labels.flatten()))
        document_embeddings = document_embeddings.reshape(labels.shape[0], labels.shape[1], self._args.dim_hidden)
        return document_labels, document_embeddings, distances, query_embedding

    def tokenize(self, query):
        tokens = self._tokenizer.tokenize(query)[:self._seq_max_len-2]
        input_tokens = [self._tokenizer.cls_token] + tokens + [self._tokenizer.sep_token]
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
        return torch.tensor(input_ids).view(1, self._seq_max_len), torch.tensor(segment_ids).view(1, self._seq_max_len), torch.tensor(input_mask).view(1, self._seq_max_len)

    def get_attention_heads(self, q_input_ids, q_input_mask, q_segment_ids):
        heads = self._model.get_attention_heads(q_input_ids, q_input_mask, q_segment_ids)
        return heads

    def load_index(self):
        logger.info('Loading KNN index...')
        self._index.load_index(self._args.index_file)

    def set_ef(self, ef):
        self._index.set_ef(ef=ef)

    def get_document(self, did, first=True):
        if self._remainP:
            if first:
                did = [self._docid2indexid[did][0]]
            else:
                did = [self._docid2indexid[did][1]]
        else:
            did = [self._docid2indexid[did]]
        return self._index.get_items(did)[0]

    def get_all_docs(self):
        ids = np.asarray(self.get_all_ids())
        docs = torch.from_numpy(np.asarray(self._index.get_items(ids)))
        doc_ids = [self._indexid2docid[index_id] for index_id in ids]
        return docs, doc_ids, torch.tensor(ids)

    def get_doc_id(self, internal_ids):
        docids = []
        for idx in internal_ids:
            docids.append(self._indexid2docid[idx.item()])
        return docids

    def get_all_ids(self):
        return self._index.get_ids_list()

    def set_device(self, device):
        self._model.to(device)

