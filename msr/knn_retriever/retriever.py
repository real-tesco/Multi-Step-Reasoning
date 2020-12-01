import hnswlib
import torch
import torch.nn as nn
import logging
import torch.optim as optim
import copy
from transformers import AutoTokenizer
import json
import numpy as np

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

        with open(args.index_mapping, 'r') as f:
            mapping = json.load(f)
        for key in mapping:
            self._indexid2docid[mapping[key]] = key
            self._docid2indexid[key] = mapping[key]

    def knn_query_text(self, query_text, k=100):
        input_ids, segment_ids, input_mask = self.tokenize(query_text)
        return self.knn_query_inference(torch.tensor([input_ids]), torch.tensor([segment_ids]), torch.tensor([input_mask]), k=k)

    def knn_query_embedded(self, query_embedding, k=100):
        query = query_embedding.detach().numpy()
        labels, distances = self._index.knn_query(query, k=k)
        distances = distances.tolist()
        #labels = labels.tolist()
        document_labels = [[self._indexid2docid[labels[j][i]] for i in range(len(labels[j]))] for j in
                           range(len(labels))]
        #labels = np.asarray(labels)
        document_embeddings = torch.tensor(self._index.get_items(labels.flatten()))
        document_embeddings = document_embeddings.reshape(labels.shape[0], labels.shape[1], self._args.dim_hidden)
        return document_labels, document_embeddings, distances, query_embedding

    def knn_query_inference(self, q_input_ids, q_segment_ids, q_input_mask, k=100):
        query_embedding = self._model.calculate_embedding(q_input_ids, q_segment_ids, q_input_mask, doc=False)
        labels, distances = self._index.knn_query(query_embedding.detach().cpu().numpy(), k=k)
        distances = distances.tolist()
        #labels = labels.tolist()
        document_labels = [[self._indexid2docid[labels[j][i]] for i in range(len(labels[j]))] for j in range(len(labels))]
        #labels = np.asarray(labels)
        document_embeddings = torch.tensor(self._index.get_items(labels.flatten()))
        document_embeddings = document_embeddings.reshape(labels.shape[0], labels.shape[1], self._args.dim_hidden)
        return document_labels, document_embeddings, distances, query_embedding

    def tokenize(self, query):
        tokens = self._tokenizer.tokenize(query)
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
        return torch.tensor(input_ids), torch.tensor(segment_ids), torch.tensor(input_mask)

    def load_index(self):
        logger.info('Loading KNN index...')
        self._index.load_index(self._args.index_file)

    def set_ef(self, ef):
        self._index.set_ef(ef=ef)

    def get_document(self, pid):
        # check if works, else pid needs to be N dim np array
        return self._index.get_items(pid)

    def set_device(self, device):
        self._model.to(device)

