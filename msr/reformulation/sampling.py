import torch
import random
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np


def cluster_sampling(documents, queries, number_samples=10, check_metrics=False):

    documents = documents.cpu().numpy()
    queries = queries.cpu().numpy()
    documents = np.concatenate((queries, documents), axis=1) # B x 1001 x 768
    # print(f"shape of documents after stack: {documents}")
    sampled_docs = torch.empty(documents.shape[0], number_samples, documents.shape[2])
    kmeans = KMeans(n_clusters=number_samples, random_state=0)
    for b in range(documents.shape[0]):
        kmeans.fit(documents[b])
        centers = torch.from_numpy(kmeans.cluster_centers_)
        sampled_docs[b] = centers
        if check_metrics:
            sil_score = silhouette_score(documents[b], kmeans.labels_, metric='cosine')
            print(f"Silhoutte Score for minibatch {b}: {sil_score}")
            if b < 4:
                print(f"query cluster lbl: {kmeans.labels_[0]}")
                sil_score_per_sample = silhouette_samples(documents[b], kmeans.labels_, metric='cosine')
                print(f"sil score of query: {sil_score_per_sample[0]}")
                # print(f"silhoutte score per sample in minibatch 0\n: {sil_score_per_sample.tolist()}")
                sil_score_per_cluster = []
                for lbl in range(number_samples):
                    sil_score_per_cluster.append(np.mean(sil_score_per_sample[kmeans.labels_ == lbl]))
                print(f'sil score per cluster: {sil_score_per_cluster}')
    return sampled_docs


def spectral_cluster_sampling(documents, number_samples=10):
    documents = documents.cpu().numpy()
    sampled_docs = torch.empty(documents.shape[0], number_samples, documents.shape[2])

    # 1. compute similarity matrix
    delta = 0.1
    sim_matrix = 1.0 - np.matmul(documents, np.transpose(documents, (0, 2, 1)))
    sim_matrix = np.exp(- sim_matrix ** 2 / (2. * delta ** 2))
    spectral_clustering = SpectralClustering(n_clusters=number_samples, random_state=0, affinity='precomputed',
                                             n_init=2)

    # 2. for every batch get clustering labels
    for b in range(documents.shape[0]):
        labels = spectral_clustering.fit_predict(sim_matrix[b])

        # 3. for each label calculate centroid
        member_count = torch.zeros(number_samples)
        for idx, lbl in enumerate(labels):
            sampled_docs[b, lbl] += documents[b, idx]
            member_count[lbl] += 1
        sampled_docs[b] = sampled_docs[b] / member_count.unsqueeze(1)

    return sampled_docs


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
