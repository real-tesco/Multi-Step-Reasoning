from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel


class TwoTowerBert(nn.Module):
    def __init__(self, pretrained: str, projection_dim=0):
        super(TwoTowerBert, self).__init__()
        self._pretrained = pretrained
        self._projection_dim = projection_dim

        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._document_model = AutoModel.from_pretrained(self._pretrained, config=self._config)
        self._query_model = AutoModel.from_pretrained(self._pretrained, config=self._config)
        if projection_dim > 0:
            self._projection_layer = nn.Linear(768, projection_dim)
            self._activation = nn.Tanh()

    def calculate_embedding(self, d_input_ids, d_input_mask, d_segment_ids, doc=True):
        if doc:
            embedding = self._document_model(d_input_ids, attention_mask=d_input_mask, token_type_ids=d_segment_ids)
        else:
            embedding = self._query_model(d_input_ids, attention_mask=d_input_mask, token_type_ids=d_segment_ids)

        rst = embedding[0][:, 0, :]

        if self._projection_dim > 0:
            rst = self._projection_layer(rst)
            rst = self._activation(rst)

        rst = F.normalize(rst, p=2, dim=1)

        return rst

    def get_attention_heads(self, q_input_ids, q_input_mask, q_segment_ids):

        output = self._query_model(q_input_ids, attention_mask=q_input_mask, token_type_ids=q_segment_ids,
                                   output_attentions=True, output_hidden_states=True,  return_dict=True)

        attention = output['attentions']   # nlayers x B x nheads x seq_len x seq_len
        hidden_states = output['hidden_states']

        last_hidden = hidden_states[-1]

        # attention -1:  torch.Size([32, 12, 64, 64])
        attention = attention[-1]
        heads = torch.empty(attention.shape[0], attention.shape[1], last_hidden.shape[2])
        for i in range(attention.shape[1]):
            # mean
            # mult = torch.bmm(attention[:, i], last_hidden).mean(dim=1)

            # only use the CLS token
            mult = torch.bmm(attention[:, i], last_hidden)[:, 0]
            mult = F.normalize(mult, p=2, dim=1)

            heads[:, i] = mult

        return heads

    def forward(self, q_input_ids: torch.Tensor, d_input_ids: torch.Tensor, q_input_mask: torch.Tensor = None,
                q_segment_ids: torch.Tensor = None, d_input_mask: torch.Tensor = None,
                d_segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        query = self._query_model(q_input_ids, attention_mask=q_input_mask, token_type_ids=q_segment_ids)
        document = self._document_model(d_input_ids, attention_mask=d_input_mask, token_type_ids=d_segment_ids)

        #print(query[0].shape)  # [4, 64, 768]
        #print(query[1].shape)  # [4, 768]  #CLS token with linear layer and tanh activation
        query = query[0][:, 0, :]  # CLS Token
        document = document[0][:, 0, :]

        if self._projection_dim > 0:
            document = self._projection_layer(document)
            document = self._activation(document)
            query = self._projection_layer(query)
            query = self._activation(query)

        document = F.normalize(document, p=2, dim=1)
        query = F.normalize(query, p=2, dim=1)

        score = (document * query).sum(dim=1)
        score = torch.clamp(score, min=0.0, max=1.0)

        return score, query, document

    def freeze_berts(self):
        self._query_model.eval()
        self._document_model.eval()

    def load_bert_model_state_dict(self, state_dict):
        st = {}
        for key in state_dict:
            if key.startswith("bert."):
                st[key[len("bert."):]] = state_dict[key]
        self._document_model.load_state_dict(st)
        self._query_model.load_state_dict(st)
