import torch
import random
from sklearn.cluster import KMeans
import numpy as np


def cluster_sampling(documents, number_samples=10):
    documents = documents.numpy()
    kmeans = KMeans(n_clusters=number_samples, random_state=0).fit(documents)
    centers = torch.from_numpy(kmeans.cluster_centers_)

    return centers


def random_sampling(documents, number_samples=10):
    """
    documents and scores have shape batch_size x K x dim_hidden

    returns sampled documents and scores with shape batch_size x number_samples x dim_hidden
    """
    sampled_docs = torch.empty(documents.shape[0], number_samples, documents.shape[2])
    for b in range(documents.shape[0]):
        tmp = random.choices(documents[b], k=number_samples)
        sampled_docs[b] = torch.stack(tmp)

    return sampled_docs


def rank_sampling(documents, number_samples=10):
    sampled_docs = torch.empty(documents.shape[0], number_samples, documents.shape[2])
    weights = [1 / (idx+1) for idx in range(documents.shape[1])]
    for b in range(documents.shape[0]):
        tmp = random.choices(documents[b], weights=weights, k=number_samples)
        sampled_docs[b] = torch.stack(tmp)

    return sampled_docs


def score_sampling(documents, scores, number_samples=10):
    sampled_docs = torch.empty(documents.shape[0], number_samples, documents.shape[2])
    for b in range(documents.shape[0]):
        weights = [score for score in scores[b]]
        tmp = random.choices(documents[b], weights=weights, k=number_samples)
        sampled_docs[b] = torch.stack(tmp)

    return sampled_docs
