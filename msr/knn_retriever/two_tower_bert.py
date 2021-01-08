from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine

from transformers import AutoConfig, AutoModel


class TwoTowerBert(nn.Module):
    def __init__(self, pretrained: str):
        super(TwoTowerBert, self).__init__()
        self._pretrained = pretrained

        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._document_model = AutoModel.from_pretrained(self._pretrained, config=self._config)
        self._query_model = AutoModel.from_pretrained(self._pretrained, config=self._config)

    def calculate_embedding(self, d_input_ids, d_input_mask, d_segment_ids, doc=True):
        if doc:
            embedding = self._document_model(d_input_ids, attention_mask=d_input_mask, token_type_ids=d_segment_ids)
        else:
            embedding = self._query_model(d_input_ids, attention_mask=d_input_mask, token_type_ids=d_segment_ids)
        rst = F.normalize(embedding[0][:, 0, :], p=2, dim=1)
        return rst

    def forward(self, q_input_ids: torch.Tensor, d_input_ids: torch.Tensor, q_input_mask: torch.Tensor = None, q_segment_ids: torch.Tensor = None,
                d_input_mask: torch.Tensor = None, d_segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        query = self._query_model(q_input_ids, attention_mask=q_input_mask, token_type_ids=q_segment_ids)
        document = self._document_model(d_input_ids, attention_mask=d_input_mask, token_type_ids=d_segment_ids)

        #print(query[0].shape)  # [4, 64, 768]
        #print(query[1].shape)  # [4, 768]  #CLS token with linear layer and tanh activation
        query = query[0][:, 0, :]  # CLS Token
        document = document[0][:, 0, :]

        #document = document - document.min()
        #query = query - query.min()
        document = F.normalize(document, p=2, dim=1)
        query = F.normalize(query, p=2, dim=1)

        score = (document * query).sum(dim=1)
        score = torch.clamp(score, min=0.0, max=1.0)

        # Use Sigmoid instead
        # score = torch.sigmoid(score)

        return score, query, document

    def load_bert_model_state_dict(self, state_dict):
        st = {}
        for key in state_dict:
            if key.startswith("bert."):
                st[key[len("bert."):]] = state_dict[key]
        self._document_model.load_state_dict(st, strict=False)
        self._query_model.load_state_dict(st, strict=False)
