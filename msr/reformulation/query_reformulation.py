import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

    def replace_with_weighted_avg(self, document_vectors, distances):
        scores = torch.ones_like(distances) - distances
        rst = self.layer.forward((document_vectors[:, :self.topk] * scores[:, :self.topk].unsqueeze(dim=-1)).sum(dim=1) / self.topk)
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
    def __init__(self, top_k, embedding_size, hidden_size1, hidden_size2):
        super(NeuralReformulator, self).__init__()
        self.top_k = top_k
        self.embedding_size = embedding_size
        self.input = nn.Linear((top_k+1)*embedding_size, hidden_size1)
        # self.h1 = nn.Linear(hidden_size1, hidden_size2)
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
    def __init__(self, topk, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=3072):
        super(TransformerReformulator, self).__init__()
        self.d_model = 768
        self.topk = topk
        self.transformer = torch.nn.Transformer(d_model=768, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers,
                                                dim_feedforward=dim_feedforward)
        self.pos_enc = PositionalEncoding(d_model=768, max_len=topk)

    # source is sequence of doc embeddings, target is correct embedding
    def forward(self, source, target):
        # source: (S, N, E) S is source sequence length here=topk, N=batchsize, E=feature number here 768
        # target: (T, N, E) T is target sequence length here=1, N and E same values as source N, E
        # needs to be transposed to match expected dimensions
        source = source[:, :self.topk].transpose(0, 1)
        target = target.unsqueeze(dim=0)
        source = self.pos_enc(source * math.sqrt(self.d_model))
        target = self.pos_enc(target * math.sqrt(self.d_model))
        return self.transformer(source, target)

    def calc_embedding(self, source):
        source = source[:, :self.topk].transpose(0, 1)
        output = self.transformer.encoder(source).squeeze(dim=0)
        return F.normalize(output, p=2, dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=768, dropout=0.1, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
