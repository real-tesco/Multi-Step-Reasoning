#!/usr/bin/python3

import torch
import torch.nn as nn


class NeuralRanker(nn.Module):
    def __init__(self, args):
        super(NeuralRanker, self).__init__()
        self.input = nn.Linear(args.ranker_input, args.ranker_hidden)
        self.h1 = nn.Linear(args.ranker_hidden, args.ranker_hidden)
        self.output = nn.Linear(args.ranker_hidden, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(args.ranker_hidden)
        self.batchnorm2 = nn.BatchNorm1d(args.ranker_hidden)

    # no sigmoid since BCE loss with logits
    def forward(self, inputs):
        x = self.input(inputs)
        x = self.batchnorm1(x)
        x = self.h1(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.output(x)

        return x

    def score_documents(self, queries, positives, negatives):
        positives = torch.cat((queries, positives), dim=1)
        scores_positive = self.forward(positives)

        negatives = torch.cat((queries, negatives), dim=1)
        scores_negative = self.forward(negatives)
        return scores_positive, scores_negative
