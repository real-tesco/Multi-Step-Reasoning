#!/usr/bin/python3

import torch
import torch.nn as nn


class NeuralRanker(nn.Module):
    def __init__(self, args):
        super(NeuralRanker, self).__init__()
        self.input = nn.Linear(args.ranker_input, args.ranker_hidden)

        if args.extra_layer > 0:
            self.extra_hidden = nn.Linear(args.ranker_hidden, args.extra_layer)
            self.h1 = nn.Linear(args.extra_layer, args.ranker_hidden)
        else:
            self.h1 = nn.Linear(args.ranker_hidden, args.ranker_hidden)
            self.extra_hidden = None

        self.output = nn.Linear(args.ranker_hidden, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.ReLU()

    # no sigmoid since BCE loss with logits
    def forward(self, inputs):
        x = self.input(inputs)
        x = self.dropout(self.activation(x))

        if self.extra_hidden is not None:
            x = self.extra_hidden(x)
            x = self.dropout(self.activation(x))

        x = self.h1(x)
        x = self.dropout(self.activation(x))

        x = self.output(x)

        return x

    def score_documents(self, queries, positives, negatives=None):

        positives = queries * positives
        scores_positive = self.forward(positives)

        if negatives is not None:
            negatives = queries * negatives
            scores_negative = self.forward(negatives)
            return scores_positive, scores_negative

        return scores_positive

    def rerank_documents(self, queries, documents, device):
        # per query K documents need to be reranked
        shape = documents.shape

        inputs = []
        for i in range(shape[1]):
            inputs.append((queries * documents[:, i, :]).tolist())
        inputs = torch.tensor(inputs).reshape(shape[0]*shape[1], shape[2]).to(device)

        scores = self.forward(inputs)
        scores = scores.reshape(shape[1], shape[0]).transpose(0, 1)

        return scores


