import torch
import random
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np


def cluster_sampling(documents, queries, qrels, document_labels, query_labels, number_samples=10, stats=None, check_metrics=False):
    document_labels = np.asarray(document_labels)
    query_labels = np.asarray(query_labels.unsqueeze(dim=1))

    documents = documents.cpu().numpy()
    queries = queries.unsqueeze(dim=1).detach().cpu().numpy()
    documents = np.concatenate((queries, documents), axis=1)    # B x 1001 x 768

    document_labels = np.concatenate((query_labels, document_labels), axis=1)

    # print(f"shape of documents after stack: {documents}")
    sampled_docs = torch.empty(documents.shape[0], number_samples, documents.shape[2])
    q_clusters = torch.empty(documents.shape[0], documents.shape[2])
    kmeans = KMeans(n_clusters=number_samples, random_state=0)
    query_sil_score = 0
    for b in range(documents.shape[0]):
        kmeans.fit(documents[b])
        centers = torch.from_numpy(kmeans.cluster_centers_)
        sampled_docs[b] = centers
        q_clusters[b] = centers[kmeans.labels_[0]]
        if check_metrics:
            sil_score = silhouette_score(documents[b], kmeans.labels_, metric='cosine')
            sil_score_per_sample = silhouette_samples(documents[b], kmeans.labels_, metric='cosine')
            # print(f"silhoutte score per sample in minibatch 0\n: {sil_score_per_sample.tolist()}")
            sil_score_per_cluster = []
            query_sil_score += sil_score_per_sample[0]

            stats['query_sil_score'] += sil_score_per_sample[0]
            stats['sil_score'] += sil_score
            stats['count'] += 1
            if sil_score_per_sample[0] > stats['sil_max']:
                stats['sil_max'] = sil_score_per_sample[0]

            if sil_score_per_sample[0] < stats['sil_min']:
                stats['sil_min'] = sil_score_per_sample[0]

            rel_docids = None
            qid = query_labels[b]
            if qid in qrels:
                rel_docids = qrels[qid]

            for lbl in range(number_samples):
                sil_score_cluster = np.mean(sil_score_per_sample[kmeans.labels_ == lbl])
                sil_score_per_cluster.append(sil_score_cluster)
                if sil_score_cluster > stats['sil_score_cluster_max']:
                    stats['sil_score_cluster_max'] = sil_score_cluster
                if sil_score_cluster < stats['sil_score_cluster_min']:
                    stats['sil_score_cluster_min'] = sil_score_cluster
                stats['sil_score_cluster'] += sil_score_cluster
                if rel_docids is not None:
                    print(f"docs: {document_labels[b].shape}")
                    docs = document_labels[b]
                    print(f"bool lbls: {(kmeans.labels_ == lbl).shape}")
                    retrieved_docs = docs[kmeans.labels_ == lbl]
                    cnt_rel_in_cluster = [[] for _ in range(number_samples)]
                    for retrieved_doc_lbl in retrieved_docs:
                        if retrieved_doc_lbl in rel_docids:
                            cnt_rel_in_cluster[lbl].append(retrieved_doc_lbl)

            if rel_docids is not None:
                print(f"rel docs (total={len(qrels[qid])})in clusters: "
                      f"{[len(cnt_rel_in_cluster[i]) for i in range(len(cnt_rel_in_cluster))]}")

            if b == 0:
                print(f"Silhoutte Score for minibatch {b}: {sil_score}")
                print(f"query cluster lbl: {kmeans.labels_[0]}")
                print(f'sil score per cluster: {sil_score_per_cluster}')
                print(f"sil score of query: {sil_score_per_sample[0]}")
    # print(f"mean query silhoutte score: {query_sil_score / documents.shape[0]}")
    return sampled_docs, q_clusters


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
