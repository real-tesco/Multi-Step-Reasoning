import argparse
from collections import defaultdict
import logging
import random
import sys

import numpy as np
from prettytable import PrettyTable
import torch
from transformers import AutoTokenizer

import msr
from msr.data.dataloader import DataLoader
from msr.data.datasets import BertDataset
from msr.data.datasets.bm25dataset import BM25Dataset
from msr.reformulation.sampling import random_sampling, score_sampling, rank_sampling, cluster_sampling, \
    spectral_cluster_sampling, attention_sampling
from msr.knn_retriever.retriever import KnnIndex
from msr.knn_retriever.retriever_config import get_args as get_knn_args
from msr.knn_retriever.two_tower_bert import TwoTowerBert
from msr.reranker.ranking_model import NeuralRanker
from msr.reranker.ranker_config import get_args as get_ranker_args
from msr.utils import Timer
from msr.reformulation.query_reformulation import QueryReformulator, TransformerReformulator, NeuralReformulator
from msr.retriever.bm25_model import BM25Retriever


logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def print_number_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def process_batch(args, rst_dict, knn_index, ranking_model, reformulator, dev_batch, device, k):
    # process batch and write result for each query in rst_dict

    second_run = True
    query_id = dev_batch['query_id']
    document_labels, document_embeddings, distances, query_embeddings = knn_index.knn_query_inference(
        dev_batch['q_input_ids'].to(device),
        dev_batch['q_input_mask'].to(device),
        dev_batch['q_segment_ids'].to(device),
        k=k)

    if not args.reformulate_before_ranking:
        batch_score = ranking_model.rerank_documents(query_embeddings, document_embeddings.to(device), device)
        # sort doc embeddings according score
        sorted_scores, scores_sorted_indices = torch.sort(batch_score, dim=1, descending=True)
        sorted_docs = document_embeddings[
            torch.arange(document_embeddings.shape[0]).unsqueeze(-1), scores_sorted_indices].to(device)
    else:
        sorted_docs = document_embeddings.to(device)
        sorted_scores = 1.0 - torch.tensor(distances)
        # print(f"batch_score max : {torch.max(sorted_scores)} min: {torch.min(sorted_scores)}")
        batch_score = sorted_scores.detach().cpu().tolist()

    if args.add_initial_retrieved:
        for (q_id, d_id, b_s) in zip(query_id, document_labels, batch_score):
            if q_id in rst_dict:
                rst_dict[q_id].extend([(docid, score) for docid, score in zip(d_id, b_s)])
            else:
                rst_dict[q_id] = [(docid, score) for docid, score in zip(d_id, b_s)]

    # do sampling regarding chosen strategy
    if args.sampling != 'none':
        if args.sampling == 'rank':
            sampled_docs = rank_sampling(sorted_docs, args.number_samples)
        elif args.sampling == 'random':
            sampled_docs = random_sampling(sorted_docs, args.number_samples)
        elif args.sampling == 'score':
            sampled_docs = score_sampling(sorted_docs, sorted_scores, args.number_samples)
        elif args.sampling == 'cluster_spectral':
            sampled_docs = spectral_cluster_sampling(sorted_docs, args.number_samples)
        else:
            raise ValueError('sampling must be `rank`, `random`, `score` or `cluster_spectral`.')

        # for each sample do the reformulation and retrieval step
        for idx in range(args.number_samples):
            '''
            TODO: remove?
            # reformulate the queries with sampled documents
            # freestyle test for transformer reformulator and sampling
            if args.reformulation_type == 'neural':
                new_queries = reformulator(query_embeddings.to(device), sampled_docs[:, :idx].unsqueeze(dim=1).to(device))
            elif args.reformulation_type == 'transformer':
                if idx == 0:
                    sampled_doc_input = sampled_docs[:, idx].unsqueeze(dim=1)
                else:
                    sampled_doc_input = sampled_docs[:, :idx]
                new_queries = reformulator(query_embeddings.to(device), sampled_doc_input.to(device))
            else:
                raise Exception(f"unsupported reformulation type for sampling: {args.reformulation_type}...")
            # freestyle test end
            '''
            new_queries = sampled_docs[:, idx]
            document_labels, document_embeddings, distances, _ = knn_index.knn_query_embedded(
                new_queries.cpu(), k=args.retrieves_per_sample)

            # batch_score = ranking_model.rerank_documents(query_embeddings, document_embeddings.to(device), device)
            sorted_scores = 1.0 - torch.tensor(distances)
            batch_score = sorted_scores
            batch_score = batch_score.detach().cpu().tolist()

            for (q_id, d_id, b_s) in zip(query_id, document_labels, batch_score):
                if q_id in rst_dict:
                    rst_dict[q_id].extend([(docid, score) for docid, score in zip(d_id, b_s)])
                else:
                    rst_dict[q_id] = [(docid, score) for docid, score in zip(d_id, b_s)]
    else:
        # reformulate the queries with top ranked documents
        if args.reformulation_type == 'neural':
            new_queries = reformulator(query_embeddings.to(device), sorted_docs)
        elif args.reformulation_type == 'weighted_avg':
            new_queries = reformulator(sorted_docs, sorted_scores.to(device))
        elif args.reformulation_type == 'transformer':
            new_queries = reformulator(query_embeddings.to(device), sorted_docs)
        elif args.reformulation_type == 'top1' or args.reformulation_type == 'top5':
            new_queries = reformulator(sorted_docs)
        else:
            # baseline
            second_run = False

        if second_run:

            # do another run with the reformulated queries

            # average over original query embedding and new calculated query embedding
            if args.avg_new_qs:
                new_queries = torch.mean(torch.stack([query_embeddings, new_queries], dim=-1), dim=-1)
                new_queries = torch.nn.functional.normalize(new_queries, p=2, dim=1)
            elif args.query_lambda > 0.0:
                new_queries = query_embeddings * args.query_lambda + (1 - args.query_lambda) * new_queries
                new_queries = torch.nn.functional.normalize(new_queries, p=2, dim=1)

            document_labels, document_embeddings, distances, _ = knn_index.knn_query_embedded(
                new_queries.cpu(), k=k)

            # rerank retrieved documents
            if args.use_ranker_in_next_round:
                batch_score = ranking_model.rerank_documents(new_queries.to(device), document_embeddings.to(device), device)
            # or use the new retrieved documents in retrieved order
            else:
                batch_score = 1.0 - torch.tensor(distances)

            batch_score = batch_score.detach().cpu().tolist()

        for (q_id, d_id, b_s) in zip(query_id, document_labels, batch_score):
            if q_id in rst_dict:
                rst_dict[q_id].extend([(docid, score) for docid, score in zip(d_id, b_s)])
            else:
                rst_dict[q_id] = [(docid, score) for docid, score in zip(d_id, b_s)]


def inference(args, knn_index, ranking_model, reformulator, dev_loader, test_loader, metric, device, k=100):
    timer = Timer()
    rst_dict_dev = {}
    rst_dict_test = {}
    if not args.skip_dev:
        logger.info("processing dev data...")
        for idx, dev_batch in enumerate(dev_loader):
            if dev_batch is None:
                continue
            process_batch(args, rst_dict_dev, knn_index, ranking_model, reformulator, dev_batch, device, k)

            if (idx+1) % args.print_every == 0:
                logger.info(f"{idx+1} / {len(dev_loader)}")
        msr.utils.save_trec_inference(args.res + ".dev", rst_dict_dev)
        logger.info(f"Time needed for {(len(dev_loader) + len(dev_loader)) * args.batch_size} examples: {timer.time()} s")
        logger.info("Eval for Dev:")
        _ = metric.eval_run(args.dev_qrels, args.res + ".dev")

    logger.info("processing test data...")
    for idx, test_batch in enumerate(test_loader):
        if test_batch is None:
            continue
        process_batch(args, rst_dict_test, knn_index, ranking_model, reformulator, test_batch, device, k)

        if (idx + 1) % args.print_every == 0:
            logger.info(f"{idx + 1} / {len(test_loader)}")
    timer.stop()

    msr.utils.save_trec_inference(args.res + ".test", rst_dict_test)

    logger.info(f"Time needed per query: {timer.time() / ((len(dev_loader) + len(test_loader)) * args.batch_size)} s")
    logger.info("Eval for Test:")
    _ = metric.eval_run(args.test_qrels, args.res + ".test")


def test_clustering(args, knn_index, ranking_model, test_loader, metric, device, k=100):
    timer = Timer()
    rst_dict_test = {}
    logger.info("processing test data...")
    stats = defaultdict(float)

    qrels = {}
    with open(args.test_qrels, "r") as f:
        for line in f:
            qid, _, did, label = line.split()
            if int(label) > 0:
                if qid in qrels:
                    qrels[qid].append(did)
                else:
                    qrels[qid] = [did]

    for idx, test_batch in enumerate(test_loader):
        if test_batch is None:
            continue

        query_id = test_batch['query_id']
        q_input_ids = test_batch['q_input_ids'].to(device)
        q_input_mask = test_batch['q_input_mask'].to(device)
        q_segment_ids = test_batch['q_segment_ids'].to(device)

        document_labels, document_embeddings, distances, query_embeddings = knn_index.knn_query_inference(
            q_input_ids,
            q_input_mask,
            q_segment_ids,
            k=k)

        if not args.reformulate_before_ranking:
            batch_score = ranking_model.rerank_documents(query_embeddings, document_embeddings.to(device), device)

            # sort doc embeddings according score
            sorted_scores, scores_sorted_indices = torch.sort(batch_score, dim=1, descending=True)
            sorted_docs = document_embeddings[
                torch.arange(document_embeddings.shape[0]).unsqueeze(-1), scores_sorted_indices].to(device)

            batch_score = batch_score.detach().cpu().tolist()

        else:
            sorted_docs = document_embeddings.to(device)
            batch_score = 1.0 - torch.tensor(distances)
            batch_score = batch_score.detach().cpu().tolist()

        # add initial retrieved set to result
        if args.add_initial_retrieved:
            for (q_id, d_id, b_s) in zip(query_id, document_labels, batch_score):
                if q_id in rst_dict_test:
                    rst_dict_test[q_id].extend([(docid, score) for docid, score in zip(d_id, b_s)])
                else:
                    rst_dict_test[q_id] = [(docid, score) for docid, score in zip(d_id, b_s)]

        # do sampling regarding chosen strategy
        if args.sampling == 'cluster_kmeans':
            check_metrics = args.print_info
            sampled_docs, q_clusters = cluster_sampling(sorted_docs, query_embeddings, args.number_samples, qrels,
                                                        document_labels, query_id, stats=stats, check_metrics=check_metrics,
                                                        print_info=args.print_info)
            # use the chosen query cluster as new query
            if args.use_q_cluster_as_q:
                new_queries = q_clusters
                tmp = args.number_samples
                args.number_samples = 1

        elif args.sampling == 'cluster_spectral':
            sampled_docs = spectral_cluster_sampling(sorted_docs, args.number_samples)

        elif args.sampling == 'attention':
            sampled_docs = attention_sampling(q_input_ids, q_input_mask, q_segment_ids, knn_index)

        for step_s in range(args.number_samples):
            if not args.use_q_cluster_as_q:
                # use sampled docs
                new_queries = sampled_docs[:, step_s]

            if args.avg_new_qs:
                new_queries = (query_embeddings + new_queries.to(device)) / 2

            document_labels, document_embeddings, distances, _ = knn_index.knn_query_embedded(
                new_queries.cpu(), k=args.retrieves_per_sample)

            if args.use_ranker_in_next_round:
                if args.avg_new_qs_for_ranking:
                    new_queries = (query_embeddings + new_queries.to(device)) / 2

                if args.rerank_to_new_qs:
                    batch_score = ranking_model.rerank_documents(new_queries.to(device), document_embeddings.to(device),
                                                                 device)
                else:
                    batch_score = ranking_model.rerank_documents(query_embeddings, document_embeddings.to(device),
                                                                 device)
            else:
                batch_score = 1.0 - torch.tensor(distances)

            batch_score = batch_score.detach().cpu().tolist()

            for (q_id, d_id, b_s) in zip(query_id, document_labels, batch_score):
                if q_id in rst_dict_test:
                    rst_dict_test[q_id].extend([(docid, score) for docid, score in zip(d_id, b_s)])
                else:
                    rst_dict_test[q_id] = [(docid, score) for docid, score in zip(d_id, b_s)]
        if args.use_q_cluster_as_q and args.sampling == 'cluster_kmeans':
            args.number_samples = tmp

        if (idx + 1) % args.print_every == 0:
            logger.info(f"{idx + 1} / {len(test_loader)}")
    timer.stop()

    # print kmeans clustering stats
    if args.sampling == 'cluster_kmeans' and args.print_info:
        for k, v in stats.items():
            print(f"{k}: {v}")
        print(f"mean_query_sil_score: {stats['query_sil_score'] / stats['count']}")
        print(f"mean_sil_score: {stats['sil_score'] / stats['count']}")
        print(f"mean_sil_score_cluster: {stats['sil_score_cluster'] / (stats['count'] * args.number_samples)}")
    msr.utils.save_trec_inference(args.res + ".test", rst_dict_test)

    logger.info(f"Time needed per query: {timer.time() / (len(test_loader) * args.batch_size)} s")
    logger.info("Eval for Test:")
    _ = metric.eval_run(args.test_qrels, args.res + ".test")
    sys.exit(0)


def eval_base_line(args):
    # evaluates the baseline with bm25 for dev and test set

    def process_base_batch(batch, rst_dict):
        query_ids = dev_batch['query_id']
        queries = dev_batch['query']
        for (qid, query) in zip(query_ids, queries):
            hits = bm25searcher.query(query, k=args.k)
            docids = [hit.docid for hit in hits]
            scores = [hit.score for hit in hits]
            for (d_id, b_s) in zip(docids, scores):
                if qid not in rst_dict_dev:
                    rst_dict_dev[qid] = [(d_id, b_s)]
                else:
                    rst_dict_dev[qid].append((d_id, b_s))

    rst_dict_dev = {}
    rst_dict_test = {}
    metric = msr.metrics.Metric()

    # DataLoaders for dev
    logger.info("Loading dev data...")
    dev_dataset = BM25Dataset(
        dataset=args.dev_data,
    )
    dev_loader = DataLoader(dev_dataset, args.batch_size, shuffle=False, num_workers=8)

    logger.info("Loading test data...")
    test_dataset = BM25Dataset(
        dataset=args.test_data,
    )
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8)
    bm25searcher = BM25Retriever(args.bm25_index)

    if args.use_rm3:
        bm25searcher.set_bm25(4.46, 0.82)
        bm25searcher.set_rm3(10, 10, 0.5)

    logger.info("Processing dev data...")
    for idx, dev_batch in enumerate(dev_loader):
        if dev_batch is None:
            continue

        process_base_batch(dev_batch, rst_dict_dev)

        if (idx + 1) % args.print_every == 0:
            logger.info(f"{idx + 1} / {len(dev_loader)}")

    logger.info("processing test data...")
    for idx, test_batch in enumerate(test_loader):
        if test_batch is None:
            continue

        process_base_batch(test_batch, rst_dict_test)

        if (idx + 1) % args.print_every == 0:
            logger.info(f"{idx + 1} / {len(test_loader)}")

    msr.utils.save_trec_inference(args.res + ".dev", rst_dict_dev)
    msr.utils.save_trec_inference(args.res + ".test", rst_dict_test)
    logger.info("Eval for Dev:")
    _ = metric.eval_run(args.dev_qrels, args.res + ".dev")
    logger.info("Eval for Test:")
    _ = metric.eval_run(args.test_qrels, args.res + ".test")
    sys.exit(0)


def eval_ideal(args, knn_index, ranking_model, device, k):
    # evaluates the ideal run, with sampling or single use of relevant documents
    def process_run_ideal(qrels, rst_dict):
        for idx, qid in enumerate(qrels):
            correct_docid = random.choice(qrels[qid])
            query = torch.tensor(knn_index.get_document(correct_docid)).unsqueeze(dim=0)

            document_labels, document_embeddings, distances, _ = knn_index.knn_query_embedded(query, k=k)

            if args.full_ranking:
                batch_score = ranking_model.rerank_documents(query.to(device), document_embeddings.to(device),
                                                             device)
                batch_score = batch_score.detach().cpu().tolist()
            else:
                batch_score = 1.0 - torch.tensor(distances)
                batch_score = batch_score.tolist()

            for (d_id, b_s) in zip(document_labels, batch_score):
                rst_dict[qid] = [(docid, score) for docid, score in zip(d_id, b_s)]

            if (idx + 1) % args.print_every == 0:
                logger.info(f"{idx + 1} / {len(qrels)}")

    def process_run_ideal_with_sampling(args, qrels, rst_dict):
        for idx, qid in enumerate(qrels):

            correct_docids = random.choices(qrels[qid], k=args.number_ideal_samples)
            query = torch.empty((args.number_ideal_samples, args.dim_embedding))
            for idy, did in enumerate(correct_docids):

                query[idy] = torch.tensor(knn_index.get_document(did))

            document_labels, document_embeddings, distances, _ = knn_index.knn_query_embedded(query, k=args.retrieves_per_sample)

            if args.full_ranking:
                batch_score = ranking_model.rerank_documents(query.to(device), document_embeddings.to(device),
                                                             device)
                # normalize score over all the queries
                #for idy in range(0, batch_score.shape[0]):
                #    batch_score[idy] = (batch_score[idy] - batch_score[idy].min()) / \
                #                       (batch_score[idy].max() - batch_score[idy].min())
                batch_score = batch_score.flatten()
                document_labels = [label for dids in document_labels for label in dids]
                batch_score = batch_score.detach().cpu().tolist()
            else:
                batch_score = 1.0 - torch.tensor(distances).flatten()
                document_labels = [label for dids in document_labels for label in dids]
                batch_score = batch_score.tolist()

            for (d_id, b_s) in zip(document_labels, batch_score):
                if qid in rst_dict:
                    rst_dict[qid].append((d_id, b_s))
                else:
                    rst_dict[qid] = [(d_id, b_s)]

            if (idx + 1) % args.print_every == 0:
                logger.info(f"{idx + 1} / {len(qrels)}")

    rst_dict_test = {}
    metric = msr.metrics.Metric()
    avg_stats = {}
    rst_dict_dev = {}
    dev_qrels = {}
    test_qrels = {}

    logger.info("Loading test data...")
    with open(args.test_qrels, "r") as f:
        for line in f:
            qid, _, docid, label = line.split()
            if int(label) > 0:
                if qid not in test_qrels:
                    test_qrels[qid] = [docid]
                else:
                    test_qrels[qid].append(docid)

    with open(args.dev_qrels, "r") as f:
        for line in f:
            qid, _, docid, label = line.split()
            if int(label) > 0:
                if qid not in dev_qrels:
                    dev_qrels[qid] = [docid]
                else:
                    dev_qrels[qid].append(docid)

    logger.info(f"len of dev qrels: {len(dev_qrels)}")
    logger.info(f"len of test qrels: {len(test_qrels)}")

    if not args.skip_dev:
        logger.info("processing dev")
        process_run_ideal(dev_qrels, rst_dict_dev)
        msr.utils.save_trec_inference(args.res + ".dev", rst_dict_dev)
        _ = metric.eval_run(args.dev_qrels, args.res + ".dev")

    logger.info("processing test")
    # since random sampling of relevant documents is random take average of number_ideal_runs
    for i in range(0, args.number_ideal_runs):
        if args.number_ideal_samples > 0:
            process_run_ideal_with_sampling(args, test_qrels, rst_dict_test)
        else:
            process_run_ideal(test_qrels, rst_dict_test)
        msr.utils.save_trec_inference(args.res + ".test", rst_dict_test)
        logger.info(f"Ideal eval for Test run #{i+1}:")
        metrics = metric.eval_run(args.test_qrels, args.res + ".test")
        for key, val in metrics.items():
            if key not in avg_stats:
                avg_stats[key] = val
            else:
                avg_stats[key] += val
        rst_dict_test = {}

    logger.info("Final result after averaging:")
    for key in avg_stats:
        avg_stats[key] = avg_stats[key] / args.number_ideal_runs
        logger.info(f"{key}:\t{avg_stats[key]}")
    sys.exit(0)


def exact_knn(args, knn_index, metric, device, k=1000):
    logger.info("loading all document embeddings from index")
    all_docs, all_docids, internal_ids = knn_index.get_all_docs()
    logger.info("load test set")
    rst_dict = {}
    test_queries = torch.from_numpy(np.load(args.test_embeddings)).to(device)
    test_q_indices = np.load(args.test_ids).tolist()

    all_docs = all_docs.to(device)

    logger.info("start large matrix multiplication...")
    first_scores = torch.matmul(test_queries.float(), torch.transpose(all_docs.float(), 0, 1))

    del all_docs
    torch.cuda.empty_cache()
    logger.info("sorting scores...")
    sorted_scores, sorted_indices = torch.sort(first_scores, dim=1, descending=True)
    sorted_internal_ids = torch.empty((sorted_scores.shape[0], k))

    logger.info("convert internal ids to docids...")
    for idx, qid in enumerate(test_q_indices):
        sorted_internal_ids[idx] = internal_ids[sorted_indices[idx]][:k]
        rst_dict[qid] = [(docid, score) for docid, score in zip(knn_index.get_doc_id(sorted_internal_ids[idx]),
                                                                sorted_scores[idx].tolist())]
    logger.info("save trec file and evaluate")
    msr.utils.save_trec_inference(args.res, rst_dict)
    metric.eval_run(args.test_qrels, args.res)
    sys.exit(0)


# generate tsv files for first 10 queries with relevant and not relevant documents
# used in https://projector.tensorflow.org/ to visualize retrieved set and query
def print_embeddings(args, knn_index):
    qrels = {}
    with open(args.test_qrels, "r") as f:
        for line in f:
            qid, _, did, label = line.split()
            # if int(label) > 0:
            if qid in qrels:
                qrels[qid].append((did, label))
            else:
                qrels[qid] = [(did, label)]

    queries = {}
    with open(args.test_embeddings, "rb") as f_emb, open(args.test_ids, "rb") as f_ind:
        qids = np.load(f_ind)
        qs = np.load(f_emb)
        for idy, qid in enumerate(qids):
            queries[qid] = qs[idy]

    for i, qid in enumerate(qrels):
        if i == 10:
            break
        with open(args.vector_file_format.format(qid), "w") as out_vector, open(args.vector_meta_format.format(qid), "w") as out_meta:
            out_meta.write('doc id\tlabel\n')
            out_meta.write(qid + '\t' + 'QUERY\n')
            out_vector.write('\t'.join([str(x) for x in queries[qid]]) + '\n')
            for did, label in qrels[qid]:
                out_vector.write('\t'.join([str(x) for x in knn_index.get_document(did)]) + '\n')
                out_meta.write(did + '\t' + str(label) + '\n')
    logger.info("finished printing vectors to files.")
    sys.exit(0)


# generate tsv files for first 3 queries with relevant and not relevant documents as well as relevant and not retrieved
# used in https://projector.tensorflow.org/ to visualize retrieved set and query and reformulated query
# also check https://projector.tensorflow.org/ for formatting
def print_reformulated_embeddings(args, knn_index, ranking_model, reformulator, device, k):
    qrels = {}
    with open(args.test_qrels, "r") as f:
        for line in f:
            qid, _, did, label = line.split()
            if qid in qrels:
                qrels[qid].append((did, label))
            else:
                qrels[qid] = [(did, label)]
    queries = {}
    if args.print_attention_sampled_embeddings:
        with open(args.test_data, "r") as qs:
            for line in qs:
                qid, query = line.split('\t')
                queries[qid] = query
    else:
        with open(args.test_embeddings, "rb") as f_emb, open(args.test_ids, "rb") as f_ind:
            qids = np.load(f_ind)
            qs = np.load(f_emb)
            for idy, qid in enumerate(qids):
                queries[qid] = qs[idy]

    # for the first 3 queries
    for idx, qid in enumerate(qrels):
        if idx == 3:
            break
        # retrieve first set
        if args.print_attention_sampled_embeddings:
            document_labels, document_embeddings, distances, original_query = knn_index.knn_query_text(queries[qid], device, k=k)
        else:
            original_query = torch.tensor(queries[qid])
            document_labels, document_embeddings, distances, _ = knn_index.knn_query_embedded(original_query, k=k)

        batch_score = ranking_model.rerank_documents(original_query.to(device), document_embeddings.to(device), device)

        sorted_scores, scores_sorted_indices = torch.sort(batch_score, dim=1, descending=True)
        sorted_docs = document_embeddings[
            torch.arange(document_embeddings.shape[0]).unsqueeze(-1), scores_sorted_indices].to(device)

        # sampled or reformulated
        if args.print_attention_sampled_embeddings:
            q_input_ids, q_input_mask, q_segment_ids = knn_index.tokenize(queries[qid])
            sampled_qs = attention_sampling(q_input_ids.to(device), q_input_mask.to(device), q_segment_ids.to(device),
                                            knn_index)[0]

            with open(args.vector_file_format.format(qid), "w") as out_vector, open(
                    args.vector_meta_format.format(qid),
                    "w") as out_meta:

                out_meta.write('doc id\tlabel\n')
                out_meta.write(qid + '\t' + 'original query\n')

                out_vector.write('\t'.join([str(x) for x in original_query[0].tolist()]) + '\n')

                judged_docs = [item[0] for item in qrels[qid]]
                labels = [item[1] for item in qrels[qid]]
                relevant_docs = [item[0] for item in qrels[qid] if int(item[1]) > 0]

                for step_s in range(args.number_samples):
                    out_meta.write(qid + f'_{step_s}' + '\t' + f'sampled query {step_s}\n')
                    out_vector.write('\t'.join([str(x) for x in sampled_qs[step_s].tolist()]) + '\n')

                    document_labels, document_embeddings, distances, _ = knn_index.knn_query_embedded(
                        sampled_qs[step_s].cpu(), k=args.retrieves_per_sample)

                    for did, doc_embed in zip(document_labels[0], document_embeddings.tolist()[0]):
                        out_vector.write('\t'.join([str(x) for x in doc_embed]) + '\n')
                        if did in judged_docs:
                            out_meta.write(did + '\t' + labels[judged_docs.index(did)] + '\n')
                        else:
                            out_meta.write(did + '\t' + 'unjudged' + '\n')
                        if did in relevant_docs:
                            relevant_docs.remove(did)
                for did in relevant_docs:
                    out_vector.write('\t'.join([str(x) for x in knn_index.get_document(did)]) + '\n')
                    out_meta.write(did + '\t' + 'relevant but not retrieved' + '\n')
        else:
            new_query = reformulator(original_query.to(device).unsqueeze(dim=0), sorted_docs)
            document_labels, document_embeddings, distances, _ = knn_index.knn_query_embedded(new_query.cpu(), k=k)

            with open(args.vector_file_format.format(qid), "w") as out_vector, open(args.vector_meta_format.format(qid),
                                                                                    "w") as out_meta:
                out_meta.write('doc id\tlabel\n')
                out_meta.write(qid + '\t' + 'original query\n')
                out_vector.write('\t'.join([str(x) for x in original_query.tolist()]) + '\n')
                out_meta.write(qid + '_' + '\t' + 'reformulated query\n')
                out_vector.write('\t'.join([str(x) for x in new_query[0].tolist()]) + '\n')

                judged_docs = [item[0] for item in qrels[qid]]
                labels = [item[1] for item in qrels[qid]]
                relevant_docs = [item[0] for item in qrels[qid] if int(item[1]) > 0]

                for did, doc_embed in zip(document_labels[0], document_embeddings.tolist()[0]):
                    out_vector.write('\t'.join([str(x) for x in doc_embed]) + '\n')
                    if did in judged_docs:
                        out_meta.write(did + '\t' + labels[judged_docs.index(did)] + '\n')
                    else:
                        out_meta.write(did + '\t' + 'unjudged' + '\n')
                    if did in relevant_docs:
                        relevant_docs.remove(did)
                for did in relevant_docs:
                    out_vector.write('\t'.join([str(x) for x in knn_index.get_document(did)]) + '\n')
                    out_meta.write(did + '\t' + 'relevant but not retrieved' + '\n')

    logger.info("finished printing vectors to files.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # reformulator args
    parser.add_argument('-reformulation_type', type=str, default=None, choices=[None, 'top1', 'top5', 'weighted_avg',
                                                                                'transformer', 'neural'])
    parser.add_argument('-reformulator_checkpoint', type=str,
                        default='./checkpoints/reformulator_transformer_loss_ip_head6_nl4.bin')
    parser.add_argument('-top_k_reformulator', type=int, default=10)
    parser.add_argument('-avg_new_qs', type='bool', default=False, help='use the avg of new query embedding and '
                                                                        'original query as input for rerun')
    parser.add_argument('-query_lambda', type=float, default=0.0, help='weighting of the original query, '
                                                                 'if > 0 original query is used')

    # transformer
    parser.add_argument('-nhead', type=int, default=6)
    parser.add_argument('-num_encoder_layers', type=int, default=4)
    parser.add_argument('-dim_feedforward', type=int, default=3072)

    # neural
    parser.add_argument('-dim_embedding', type=int, default=768)
    parser.add_argument('-hidden1', type=int, default=2500)
    parser.add_argument('-hidden2', type=int, default=0)

    # sampling
    parser.add_argument('-sampling', type=str, default='none', choices=['none', 'rank', 'score', 'random',
                                                                        'cluster_kmeans', 'cluster_spectral',
                                                                        'attention'],
                        help='type of sampling to use before reformulation, default is none, just use top documents')

    # clustering settings
    parser.add_argument('-retrieves_per_sample', type=int, default=100, help='the number of retrieves per sample')
    parser.add_argument('-number_samples', type=int, default=10, help='the number of samples per query')
    parser.add_argument('-test_clustering', type='bool', default=False,
                        help='test clustering and eval clustering metrics')
    parser.add_argument('-use_q_cluster_as_q', type='bool', default=False,
                        help='in test clustering use the q cluster centroid as query representation')
    parser.add_argument('-avg_new_qs_for_ranking', type='bool', default=False,
                        help='average the new queries and the original queries for ranking')
    parser.add_argument('-rerank_to_new_qs', type='bool', default=False, help='rerank to new queries')
    parser.add_argument('-print_info', type='bool', default=False, help='print clustering info in every step')
    parser.add_argument('-add_initial_retrieved', type='bool', default=False,
                        help='whether to add the first step retrieval to result')

    # run options
    parser.add_argument('-baseline', type='bool', default='False', help="if true only use bm25 to score documents")
    parser.add_argument('-print_embeddings', type='bool', default=False)
    parser.add_argument('-print_reformulated_embeddings', type='bool', default=False)
    parser.add_argument('-print_attention_sampled_embeddings', type='bool', default=False)
    parser.add_argument('-ideal', type='bool', default='False', help='wether use correct doc embeddings as queries')
    parser.add_argument('-number_ideal_runs', type=int, default=10)
    parser.add_argument('-number_ideal_samples', type=int, default=0)
    parser.add_argument('-exact_knn', type='bool', default=False)

    # print options
    parser.add_argument('-print_model_params', type='bool', default=False)
    parser.add_argument('-vector_file_format', type=str,
                        default='./data/embeddings/embeddings_random_examples/qid_{}_judged_docs.tsv')
    parser.add_argument('-vector_meta_format', type=str,
                        default='./data/embeddings/embeddings_random_examples/qid_{}_meta.tsv')

    parser.add_argument('-metric', type=str, default='mrr_cut_100')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-print_every', type=int, default=25)
    parser.add_argument('-train', type='bool', default=False)
    parser.add_argument('-bm25_index', type=str, default='./data/indexes/anserini/index-msmarco-doc-20201117-f87c94')
    parser.add_argument('-use_rm3', type='bool', default=False)
    parser.add_argument('-skip_dev', type='bool', default=False)

    # data options
    parser.add_argument('-res', type=str, default='./results/twotowerbert.trec')
    parser.add_argument('-two_tower_checkpoint', type=str, default='./checkpoints/twotowerbert.bin')
    parser.add_argument('-ranker_checkpoint', type=str, default='./checkpoints/ranker_extra_layer_2500.ckpt')

    parser.add_argument('-dev_data', action=msr.utils.DictOrStr, default='./data/msmarco-dev-queries-inference.jsonl')
    parser.add_argument('-dev_qrels', type=str, default='./data/msmarco-docdev-qrels.tsv')
    parser.add_argument('-test_data', action=msr.utils.DictOrStr, default='./data/msmarco-test-queries-inference.jsonl')
    parser.add_argument('-test_qrels', type=str, default='./data/msmarco-test-qrels.tsv')
    parser.add_argument('-test_embeddings', type=str,
                        default='./data/embeddings/embeddings_random_examples/marco_test_query_embeddings_0.npy')
    parser.add_argument('-test_ids', type=str,
                        default='./data/embeddings/embeddings_random_examples/marco_test_query_embeddings_indices_0.npy')
    parser.add_argument('-doc_emb_format', type=str,
                        default='./data/embeddings/embeddings_random_examplesmarco_doc_embeddings_{}.npy')
    parser.add_argument('-doc_ids_format', type=str,
                        default='./data/embeddings/embeddings_random_examplesmarco_doc_embeddings_indices_{}.npy')
    parser.add_argument('-num_doc_files', type=int, default=13)

    # pipeline options
    parser.add_argument('-k', type=int, default=100)
    parser.add_argument('-full_ranking', type='bool', default=False)
    parser.add_argument('-reformulate_before_ranking', type='bool', default=True,
                        help='if true dont use ranker before reformulation')
    parser.add_argument('-use_ranker_in_next_round', type='bool', default=False,
                        help='whether to use ranker in next round or not')

    index_args = get_knn_args(parser)
    ranker_args = get_ranker_args(parser)
    args = parser.parse_args()

    if args.baseline:
        eval_base_line(args)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set metric
    metric = msr.metrics.Metric()

    logger.info("Loading Retriever Model...")
    two_tower_bert = TwoTowerBert(index_args.pretrain)
    checkpoint = torch.load(args.two_tower_checkpoint)
    two_tower_bert.load_state_dict(checkpoint)
    two_tower_bert.eval()

    knn_index = KnnIndex(index_args, two_tower_bert)
    logger.info("Load Index File and set ef...")
    knn_index.load_index()
    knn_index.set_ef(index_args.efc)
    knn_index.set_device(device)

    if args.exact_knn:
        logger.info("Start calculating exact knn")
        exact_knn(args, knn_index, metric, device, k=1000)
    if args.print_embeddings:
        logger.info("Start printing vectors to files...")
        print_embeddings(args, knn_index)

    if args.reformulation_type is not None:
        logger.info('Loading Reformulator...')
        checkpoint = torch.load(args.reformulator_checkpoint)
        if args.reformulation_type == 'neural':
            reformulator = NeuralReformulator(args.top_k_reformulator, args.dim_embedding, args.hidden1, args.hidden2)
            reformulator.load_state_dict(checkpoint)
            reformulator.to(device)
            reformulator.eval()
            if args.print_model_params:
                print_number_parameters(reformulator)

        elif args.reformulation_type == 'weighted_avg':
            reformulator = QueryReformulator(mode='weighted_avg', topk=args.top_k_reformulator)
            reformulator.layer.load_state_dict(checkpoint)
            reformulator.layer.to(device)
            reformulator.layer.eval()
            if args.print_model_params:
                print_number_parameters(reformulator.layer)

        elif args.reformulation_type == 'transformer':
            reformulator = TransformerReformulator(args.top_k_reformulator, args.nhead, args.num_encoder_layers,
                                                   args.dim_feedforward)
            # maybe some issues regarding used pytorch version
            # reformulator.load_state_dict(checkpoint)
            reformulator.load_fixed_checkpoint(args.reformulator_checkpoint)
            reformulator.to_device(device)
            reformulator.eval()
            if args.print_model_params:
                print_number_parameters(reformulator)

        elif args.reformulation_type == 'top1':
            reformulator = QueryReformulator(mode='top1')
        elif args.reformulation_type == 'top5':
            reformulator = QueryReformulator(mode='top5', topk=5)
    else:
        reformulator = None

    if args.full_ranking or args.use_ranker_in_next_round or not args.reformulate_before_ranking:
        # Load Ranker
        logger.info("Loading Ranker...")
        ranking_model = NeuralRanker(ranker_args)
        checkpoint = torch.load(args.ranker_checkpoint)
        ranking_model.load_state_dict(checkpoint)
        ranking_model.to(device)
        ranking_model.eval()
    else:
        logger.info("No ranker is used...")
        ranking_model = None

    if args.ideal:
        eval_ideal(args, knn_index, ranking_model, device, k=args.k)

    if args.print_reformulated_embeddings or args.print_attention_sampled_embeddings:
        print_reformulated_embeddings(args, knn_index, ranking_model, reformulator, device, k=args.k)

    # DataLoaders for dev
    logger.info("Loading dev data...")
    tokenizer = AutoTokenizer.from_pretrained(index_args.pretrain)
    dev_dataset = BertDataset(
        dataset=args.dev_data,
        tokenizer=tokenizer,
        mode='inference',
        query_max_len=index_args.max_query_len,
        doc_max_len=index_args.max_doc_len,
        max_input=args.max_input
    )
    dev_loader = DataLoader(dev_dataset, args.batch_size, shuffle=False, num_workers=8)

    logger.info("Loading test data...")
    test_dataset = BertDataset(
        dataset=args.test_data,
        tokenizer=tokenizer,
        mode='inference',
        query_max_len=index_args.max_query_len,
        doc_max_len=index_args.max_doc_len,
        max_input=args.max_input
    )
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8)

    if args.sampling == 'attention':
        args.number_samples = args.nhead

    if args.test_clustering:
        logger.info("Start cluster testing...")
        test_clustering(args, knn_index, ranking_model, test_loader, metric, device, args.k)

    # starting inference
    logger.info("Starting inference...")
    inference(args, knn_index, ranking_model, reformulator, dev_loader, test_loader, metric, device, args.k)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    file_handler = logging.FileHandler('./train_retriever.log', mode='w', encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    main()
