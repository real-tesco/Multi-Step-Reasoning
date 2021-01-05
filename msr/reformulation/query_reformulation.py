import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
import math
import copy


class QueryReformulator:
    def __init__(self, mode: str, topk=None):
        self._mode = mode
        if mode == 'weighted_avg':
            self.layer = ProjectionLayer(dim_input=768, dim_output=768, mode='single')
        if topk is not None:
            self.topk = topk

    def __call__(self, *args, **kwargs):
        if self._mode == 'top1':
            return self.replace_with_document(*args)
        elif self._mode == 'top5':
            return self.replace_with_avg(*args)
        elif self._mode == 'weighted_avg':
            return self.replace_with_weighted_avg(*args)

    def replace_with_document(self, document_vectors):
        return document_vectors[:, 0]

    def replace_with_avg(self, document_vectors):
        rst = torch.mean(document_vectors[:, :self.topk], dim=1)
        return rst

    def replace_with_weighted_avg(self, document_vectors, scores):
        rst = self.layer.forward((document_vectors[:, :self.topk] * scores[:, :self.topk].unsqueeze(dim=-1)).sum(dim=1)
                                 / scores[:, :self.topk].unsqueeze(dim=-1).sum(dim=1))
        rst = F.normalize(rst, p=2, dim=1)
        return rst


class ProjectionLayer(nn.Module):
    def __init__(self, dim_input, dim_output=768, mode='ip'):
        super(ProjectionLayer, self).__init__()
        self._mode = mode
        self._layer = nn.Linear(dim_input, dim_output)

    # input as inner product, as concatenated, single vector input
    def forward(self, query_embedding, document_embedding=None):
        # inner product
        if self._mode == 'ip':
            inputs = query_embedding * document_embedding
        elif self._mode == 'cat':
            inputs = torch.cat([query_embedding, document_embedding])
        else:
            inputs = query_embedding
        return self._layer(inputs)


class NeuralReformulator(nn.Module):
    def __init__(self, top_k, embedding_size, hidden_size1):
        super(NeuralReformulator, self).__init__()
        self.top_k = top_k
        self.embedding_size = embedding_size
        self.input = nn.Linear((top_k+1)*embedding_size, hidden_size1)
        self.output = nn.Linear(hidden_size1, embedding_size)
        self.activation = nn.Sigmoid()

    def forward(self, query_embedding, document_embeddings):
        if len(query_embedding.shape) == 1:
            inputs = torch.cat([torch.unsqueeze(query_embedding, dim=0).t(), document_embeddings[:self.top_k].t()], dim=1).flatten()
        else:
            q_emb = torch.unsqueeze(query_embedding, dim=2)
            d_emb = document_embeddings[:, :self.top_k].transpose(1, 2)
            inputs = torch.cat([q_emb, d_emb], dim=2)
            inputs = inputs.flatten(start_dim=1)

        #print(inputs.shape)
        x = self.activation(self.input(inputs))
        # x = self.activation(self.h1(x))
        x = self.output(x)

        if len(query_embedding.shape) == 1:
            x = F.normalize(x, p=2, dim=0)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x


class TransformerReformulator(nn.Module):
    def __init__(self, topk, nhead=4, num_encoder_layers=1, dim_feedforward=3072):
        super(TransformerReformulator, self).__init__()
        self.d_model = 768
        self.topk = topk

        self.pos_enc = PositionalEncoding(d_model=768, max_len=topk + 1)   # query on index 0
        encoder_layer = TransformerEncoderLayer(d_model=768, nhead=nhead, dim_feedforward=dim_feedforward)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

        self.decoder = nn.Linear(768, 768)

    def forward(self, query, source_embeddings):
        # source_embeddings: (S, N, E) S is source sequence length here=topk, N=batchsize, E=feature number here 768
        # query: (N, E) N and E same values as source N, E
        # needs to be transposed to match expected dimensions
        source = source_embeddings[:, :self.topk].transpose(0, 1)
        query = query.unsqueeze(dim=0)
        source = torch.cat([query, source])
        source = self.pos_enc(source * math.sqrt(self.d_model))
        output = source
        for layer in self.layers:
            output = layer(output)
        # output at index 0 is the query representation
        output = output[0, :]
        output = nn.functional.normalize(output, p=2, dim=1)
        return output

    # refactor needed
    # hack to not retrain the reformulators
    def load_fixed_checkpoint(self, path):
        m = torch.load(path)
        model_dict = self.state_dict()
        for k in m.keys():
            if '"pos_enc.pe"' in k:
                continue
            if k in model_dict:
                pname = k
                pval = m[k]
                model_dict[pname] = pval.clone().to(model_dict[pname].device)
        self.load_state_dict(model_dict)

    def to_device(self, device):
        self.to(device)
        self.pos_enc.pe = self.pos_enc.pe.to(device)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=768, dropout=0.1, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = self.pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x
