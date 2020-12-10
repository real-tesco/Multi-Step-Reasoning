import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryReformulator:
    def __init__(self, mode: str):
        self._mode = mode

    def __call__(self, *args, **kwargs):
        if self._mode == 'top1':
            return self.replace_with_document(*args)
        elif self._mode == 'top5':
            return self.replace_with_avg(*args)

    def replace_with_document(self, document_vectors):
        return document_vectors[:, 0]

    def replace_with_avg(self, document_vectors):
        rst = torch.mean(document_vectors[:, :5], dim=1)
        return rst


class NeuralReformulator(nn.Module):
    def __init__(self, top_k, embedding_size, hidden_size1, hidden_size2):
        super(NeuralReformulator, self).__init__()
        self.top_k = top_k
        self.embedding_size = embedding_size
        self.input = nn.Linear((top_k+1)*embedding_size, hidden_size1)
        self.h1 = nn.Linear(hidden_size1, hidden_size2)
        self.output = nn.Linear(hidden_size2, embedding_size)
        self.activation = nn.SiLU()

    def forward(self, query_embedding, document_embeddings):
        if len(query_embedding.shape) == 1:
            inputs = torch.cat([torch.unsqueeze(query_embedding, dim=0).t(), document_embeddings[:self.top_k].t()], dim=1).flatten()
        else:
            q_emb = torch.unsqueeze(query_embedding, dim=2)
            d_emb = document_embeddings[:, :self.top_k].transpose(1, 2)
            inputs = torch.cat([q_emb, d_emb], dim=2)
            inputs = inputs.flatten(start_dim=1)
        #print(inputs.shape)
        x = self.input(inputs)
        x = self.activation(self.h1(x))
        x = self.activation(self.output(x))
        return x

