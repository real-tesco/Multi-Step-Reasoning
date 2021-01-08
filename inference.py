import argparse
import torch
import msr
from msr.data.dataloader import DataLoader
from msr.data.datasets import BertDataset
from msr.data.datasets.rankingdataset import RankingDataset
from transformers import AutoTokenizer
from msr.knn_retriever.retriever import KnnIndex
from msr.knn_retriever.retriever_config import get_args as get_knn_args
from msr.knn_retriever.two_tower_bert import TwoTowerBert
from msr.reranker.ranking_model import NeuralRanker
from msr.reranker.ranker_config import get_args as get_ranker_args
from msr.reformulation.reformulator_config import get_args as get_reformulator_args
from msr.utils import Timer
from msr.reformulation.query_reformulation import QueryReformulator, TransformerReformulator, NeuralReformulator
import logging
import numpy as np
from msr.retriever.bm25_model import BM25Retriever
from msr.data.datasets.bm25dataset import BM25Dataset
import random

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def process_batch(args, rst_dict, knn_index, ranking_model, reformulator, dev_batch, device, k):
    second_run = True
    query_id = dev_batch['query_id']
    document_labels, document_embeddings, distances, query_embeddings = knn_index.knn_query_inference(
        dev_batch['q_input_ids'].to(device),
        dev_batch['q_input_mask'].to(device),
        dev_batch['q_segment_ids'].to(device),
        k=k)

    batch_score = ranking_model.rerank_documents(query_embeddings.to(device), document_embeddings.to(device), device)

    # sort doc embeddings according score and reformulate
    sorted_scores, scores_sorted_indices = torch.sort(batch_score, dim=1, descending=True)
    sorted_docs = document_embeddings[
        torch.arange(document_embeddings.shape[0]).unsqueeze(-1), scores_sorted_indices].to(device)

    if args.reformulation_type == 'neural':
        new_queries = reformulator(query_embeddings.to(device), sorted_docs)
    elif args.reformulation_type == 'weighted_avg':
        new_queries = reformulator(sorted_docs, sorted_scores.to(device))
    elif args.reformulation_type == 'transformer':
        new_queries = reformulator(query_embeddings.to(device), sorted_docs)
    else:
        # baseline
        second_run = False

    if second_run:

        # do another run with the reformulated queries

        # average over original query embedding and new calculated query embedding
        if args.avg_new_qs:
            new_queries = torch.mean(torch.stack([query_embeddings, new_queries], dim=-1), dim=-1)

        document_labels, document_embeddings, distances, _ = knn_index.knn_query_embedded(
            new_queries.cpu())

        batch_score = ranking_model.rerank_documents(new_queries.to(device), document_embeddings.to(device),
                                                     device)
    batch_score = batch_score.detach().cpu().tolist()

    for (q_id, d_id, b_s) in zip(query_id, document_labels, batch_score):
        rst_dict[q_id] = [(docid, score) for docid, score in zip(d_id, b_s)]


def inference(args, knn_index, ranking_model, reformulator, dev_loader, test_loader, metric, device, k=100):
    timer = Timer()
    rst_dict_dev = {}
    rst_dict_test = {}
    logger.info("processing dev data...")
    for idx, dev_batch in enumerate(dev_loader):
        if dev_batch is None:
            continue
        process_batch(args, rst_dict_dev, knn_index, ranking_model, reformulator, dev_batch, device, k)

        if (idx+1) % args.print_every == 0:
            logger.info(f"{idx+1} / {len(dev_loader)}")

    logger.info("processing test data...")
    for idx, test_batch in enumerate(test_loader):
        if test_batch is None:
            continue
        process_batch(args, rst_dict_test, knn_index, ranking_model, reformulator, test_batch, device, k)

        if (idx + 1) % args.print_every == 0:
            logger.info(f"{idx + 1} / {len(test_loader)}")
    timer.stop()
    msr.utils.save_trec_inference(args.res + ".dev", rst_dict_dev)
    msr.utils.save_trec_inference(args.res + ".test", rst_dict_test)

    logger.info(f"Time needed for {(len(dev_loader) + len(dev_loader)) * args.batch_size} examples: {timer.time()} s")
    logger.info(f"Time needed per query: {timer.time() / ((len(dev_loader) + len(test_loader)) * args.batch_size)} s")
    logger.info("Eval for Dev:")
    _ = metric.eval_run(args.dev_qrels, args.res + ".dev")
    logger.info("Eval for Test:")
    _ = metric.eval_run(args.test_qrels, args.res + ".test")


def eval_base_line(args):
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

    logger.info("Processing dev data...")
    for idx, dev_batch in enumerate(dev_loader):
        if dev_batch is None:
            continue
        query_ids = dev_batch['query_id']
        queries = dev_batch['query']
        for (qid, query) in zip(query_ids, queries):
            hits = bm25searcher.query(query, k=100)
            docids = [hit.docid for hit in hits]
            scores = [hit.score for hit in hits]
            for (d_id, b_s) in zip(docids, scores):
                if qid not in rst_dict_dev:
                    rst_dict_dev[qid] = [(d_id, b_s)]
                else:
                    rst_dict_dev[qid].append((d_id, b_s))
        if (idx + 1) % args.print_every == 0:
            logger.info(f"{idx + 1} / {len(dev_loader)}")

    logger.info("processing test data...")
    for idx, test_batch in enumerate(test_loader):
        if test_batch is None:
            continue
        query_ids = test_batch['query_id']
        queries = test_batch['query']
        for (qid, query) in zip(query_ids, queries):
            hits = bm25searcher.query(query, k=100)
            docids = [hit.docid for hit in hits]
            scores = [hit.score for hit in hits]
            for (d_id, b_s) in zip(docids, scores):
                if qid not in rst_dict_test:
                    rst_dict_test[qid] = [(d_id, b_s)]
                else:
                    rst_dict_test[qid].append((d_id, b_s))
        if (idx + 1) % args.print_every == 0:
            logger.info(f"{idx + 1} / {len(test_loader)}")
    msr.utils.save_trec_inference(args.res + ".dev", rst_dict_dev)
    msr.utils.save_trec_inference(args.res + ".test", rst_dict_test)
    logger.info("Eval for Dev:")
    _ = metric.eval_run(args.dev_qrels, args.res + ".dev")
    logger.info("Eval for Test:")
    _ = metric.eval_run(args.test_qrels, args.res + ".test")
    exit(0)


def eval_ideal(args, knn_index, ranking_model, device):
    def process_run_ideal(qrels, rst_dict):
        for idx, qid in enumerate(qrels):
            correct_docid = random.choice(qrels[qid])
            query = torch.tensor(knn_index.get_document(correct_docid)).unsqueeze(dim=0)

            document_labels, document_embeddings, distances, _ = knn_index.knn_query_embedded(query)

            if args.full_ranking:
                batch_score = ranking_model.rerank_documents(query.to(device), document_embeddings.to(device),
                                                             device)
                batch_score = batch_score.detach().cpu().tolist()
            else:
                batch_score = distances

            for (d_id, b_s) in zip(document_labels, batch_score):
                rst_dict[qid] = [(docid, score) for docid, score in zip(d_id, b_s)]

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

    logger.info("processing dev")
    process_run_ideal(dev_qrels, rst_dict_dev)
    msr.utils.save_trec_inference(args.res + ".dev", rst_dict_dev)
    _ = metric.eval_run(args.test_qrels, args.res + ".dev")

    logger.info("processing test")
    for i in range(0, args.number_ideal_runs):
        process_run_ideal(test_qrels, rst_dict_test)
        msr.utils.save_trec_inference(args.res + ".test", rst_dict_test)
        logger.info(f"Ideal eval for Test run #{i+1}:")
        metrics = metric.eval_run(args.test_qrels, args.res + ".test")
        for key, val in metrics.items():
            if key not in avg_stats:
                avg_stats[key] = val
            else:
                avg_stats[key] += val

    logger.info("Final result after averaging:")
    for key in avg_stats:
        avg_stats[key] = avg_stats[key] / args.number_ideal_runs
        logger.info(f"{key}:\t{avg_stats[key]}")
    exit(0)


def main():
    # setting args
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # reformulator args
    parser.add_argument('-reformulation_type', type=str, default=None, choices=[None, 'top1', 'top5', 'weighted_avg',
                                                                                'transformer', 'neural'])
    parser.add_argument('-reformulator_checkpoint', type=str, default='./checkpoints/reformulator_transformer_loss_ip_lr_top10.bin')
    parser.add_argument('-top_k_reformulator', type=int, default=10)
    parser.add_argument('-avg_new_qs', type='bool', default=False, help='use the avg of new query embedding and original '
                                                                        'query as input for rerun')

    # transformer
    parser.add_argument('-nhead', type=int, default=6)
    parser.add_argument('-num_encoder_layers', type=int, default=4)
    parser.add_argument('-dim_feedforward', type=int, default=3072)

    # neural
    parser.add_argument('-dim_embedding', type=int, default=768)
    parser.add_argument('-hidden1', type=int, default=2500)

    parser.add_argument('-baseline', type='bool', default='False', help="if true only use bm25 to score documents")
    parser.add_argument('-ideal', type='bool', default='False', help='wether use correct doc embeddings as queries')
    parser.add_argument('-number_ideal_runs', type=int, default=10)
    parser.add_argument('-bm25_index', type=str, default='./data/indexes/anserini/index-msmarco-doc-20201117-f87c94')

    parser.add_argument('-two_tower_checkpoint', type=str, default='./checkpoints/twotowerbert.bin')
    parser.add_argument('-ranker_checkpoint', type=str, default='./checkpoints/ranker_extra_layer_2500.ckpt')
    parser.add_argument('-dev_data', action=msr.utils.DictOrStr, default='./data/msmarco-dev-queries-inference.jsonl')
    parser.add_argument('-dev_qrels', type=str, default='./data/msmarco-docdev-qrels.tsv')

    parser.add_argument('-test_data', action=msr.utils.DictOrStr, default='./data/msmarco-test-queries-inference.jsonl')
    parser.add_argument('-test_qrels', type=str, default='./data/msmarco-test-qrels.tsv')

    parser.add_argument('-res', type=str, default='./results/twotowerbert.trec')
    parser.add_argument('-metric', type=str, default='mrr_cut_100')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-print_every', type=int, default=25)
    parser.add_argument('-train', type='bool', default=False)
    parser.add_argument('-full_ranking', type='bool', default=True)

    parser.add_argument('-k', type=int, default=100)
    parser.add_argument('-use_ranker_in_next_round', type='bool', default=True)

    args = parser.parse_args()
    # re_args = get_reformulator_args(parser)
    index_args = get_knn_args(parser)
    ranker_args = get_ranker_args(parser)
    ranker_args.train = False

    if args.baseline:
        eval_base_line(args)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #________________TEST__________
    if args.full_ranking or args.use_ranker_in_next_round:
        #   2. Load Ranker
        logger.info("Loading Ranker...")
        #ranker_args = get_ranker_args(parser)
        ranking_model = NeuralRanker(ranker_args)
        checkpoint = torch.load(args.ranker_checkpoint)
        ranking_model.load_state_dict(checkpoint)
        ranking_model.eval()
        ranking_model.to(device)
    else:
        logger.info("No ranker is used...")
        ranking_model = None


    # Loading models
    #    1. Load Retriever
    logger.info("Loading Retriever...")
    two_tower_bert = TwoTowerBert(index_args.pretrain)
    checkpoint = torch.load(args.two_tower_checkpoint)
    two_tower_bert.load_state_dict(checkpoint, strict=False)
    two_tower_bert.eval()
    knn_index = KnnIndex(index_args, two_tower_bert)
    logger.info("Load Index File and set ef")
    knn_index.load_index()
    knn_index.set_ef(index_args.efc)
    knn_index.set_device(device)

    if args.reformulation_type is not None:
        logger.info('Loading Reformulator...')
        checkpoint = torch.load(args.reformulator_checkpoint)
        if args.reformulation_type == 'neural':
            reformulator = NeuralReformulator(args.top_k_reformulator, args.dim_embedding, args.hidden1)
            reformulator.load_state_dict(checkpoint)
            reformulator.to(device)
            reformulator.eval()
        elif args.reformulation_type == 'weighted_avg':
            reformulator = QueryReformulator(mode='weighted_avg', topk=args.top_k_reformulator)
            reformulator.layer.load_state_dict(checkpoint)
            reformulator.layer.to(device)
            reformulator.layer.eval()
        elif args.reformulation_type == 'transformer':
            reformulator = TransformerReformulator(args.top_k_reformulator, args.nhead, args.num_encoder_layers,
                                                   args.dim_feedforward)
            # reformulator.load_state_dict(checkpoint)
            reformulator.load_fixed_checkpoint(args.reformulator_checkpoint)
            reformulator.to_device(device)
            reformulator.eval()
    else:
        reformulator = None

    if args.full_ranking or args.use_ranker_in_next_round:
        #   2. Load Ranker
        logger.info("Loading Ranker...")
        #ranker_args = get_ranker_args(parser)
        ranking_model = NeuralRanker(ranker_args)
        checkpoint = torch.load(args.ranker_checkpoint)
        ranking_model.load_state_dict(checkpoint)
        ranking_model.to(device)
        ranking_model.eval()
    else:
        logger.info("No ranker is used...")
        ranking_model = None

    if args.ideal:
        eval_ideal(args, knn_index, ranking_model, device)

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

    # set metric
    metric = msr.metrics.Metric()

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
