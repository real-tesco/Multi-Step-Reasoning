import argparse
import torch
import msr
from msr.data.dataloader import DataLoader
from msr.data.datasets import BertDataset
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


logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def process_batch(args, rst_dict_dev, knn_index, ranking_model, reformulator, dev_batch, device, k):

    query_id = dev_batch['query_id']
    document_labels, document_embeddings, distances, query_embeddings = knn_index.knn_query_inference(
        dev_batch['q_input_ids'].to(device),
        dev_batch['q_input_mask'].to(device),
        dev_batch['q_segment_ids'].to(device),
        k=k)

    if args.full_ranking:
        batch_score = ranking_model.rerank_documents(query_embeddings.to(device), document_embeddings.to(device),
                                                     device)
        batch_score = batch_score.detach().cpu().tolist()
    else:
        batch_score = distances

    if args.reformulation_mode:
        # sort doc embeddings according score and reformulate
        _, scores_sorted_indices = torch.sort(torch.tensor(batch_score), dim=1, descending=True)
        sorted_docs = document_embeddings[
            torch.arange(document_embeddings.shape[0]).unsqueeze(-1), scores_sorted_indices]
        new_queries = reformulator(sorted_docs)

        # retrieve new set of candidate documents
        document_labels, document_embeddings, distances, query_embeddings = knn_index.knn_query_embedded(
            new_queries, k=k)

        # rerank
        if args.use_ranker_in_next_round:
            batch_score = ranking_model.rerank_documents(query_embeddings.to(device),
                                                         document_embeddings.to(device), device)
            batch_score = batch_score.detach().cpu().tolist()
        else:
            batch_score = distances

    for (q_id, d_id, b_s) in zip(query_id, document_labels, batch_score):
        rst_dict_dev[q_id] = [(score, docid) for score, docid in zip(d_id, b_s)]


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
            logger.info(f"{idx + 1} / {len(dev_loader)}")
    timer.stop()
    msr.utils.save_trec_inference(args.res + ".dev", rst_dict_dev)
    msr.utils.save_trec_inference(args.res + ".test", rst_dict_test)

    logger.info(f"Time needed for {(len(dev_loader) + len(dev_loader)) * args.batch_size} examples: {timer.time()} s")
    logger.info(f"Time needed per query: {timer.time() / ((len(dev_loader) + len(test_loader)) * args.batch_size)} s")
    logger.info("Eval for Dev:")
    _ = metric.eval_run(args.qrels, args.res + ".dev")
    logger.info("Eval for Test:")
    _ = metric.eval_run(args.qrels, args.res + ".test")


def main():
    # setting args
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # reformulator args
    parser.add_argument('-reformulation_mode', type=str, default=None, choices=[None, 'top1', 'top5', 'weighted_avg',
                                                                                'transformer'])
    parser.add_argument('-reformulator_checkpoint', type=str, default='./checkpoints/reformulator_transformer_loss_ip_lr_top10.bin')

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
    re_args = get_reformulator_args(parser)
    index_args = get_knn_args(parser)
    ranker_args = get_ranker_args(parser)
    ranker_args.train = False

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

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading models
    #    1. Load Retriever
    logger.info("Loading Retriever...")
    two_tower_bert = TwoTowerBert(index_args.pretrain)
    checkpoint = torch.load(args.two_tower_checkpoint)
    two_tower_bert.load_state_dict(checkpoint)
    knn_index = KnnIndex(index_args, two_tower_bert)
    logger.info("Load Index File and set ef")
    knn_index.load_index()
    knn_index.set_ef(index_args.efc)
    knn_index.set_device(device)

    logger.info('Loading Reformulator...')
    checkpoint = torch.load(args.reformulator_checkpoint)
    if args.reformulation_mode == 'neural':
        reformulator = NeuralReformulator(re_args.top_k_reformulator, re_args.dim_embedding, re_args.hidden1)
        reformulator.load_state_dict(checkpoint)
        reformulator.to(device)
    elif args.reformulation_mode == 'weighted_avg':
        reformulator = QueryReformulator(mode='weighted_avg', topk=re_args.top_k_reformulator)
        reformulator.layer.load_state_dict(checkpoint)
        reformulator.layer.to(device)
    elif args.reformulation_mode == 'transformer':
        reformulator = TransformerReformulator(re_args.top_k_reformulator, re_args.nhead, re_args.num_encoder_layers,
                                               re_args.dim_feedforward)
        reformulator.load_state_dict(checkpoint)
        reformulator.to(device)
        if torch.cuda.device_count() > 1:
            logger.info(f'Using DataParallel with {torch.cuda.device_count()} GPUs...')
            reformulator = torch.nn.DataParallel(reformulator)
    else:
        return

    if args.full_ranking or args.use_ranker_in_next_round:
        #   2. Load Ranker
        logger.info("Loading Ranker...")
        #ranker_args = get_ranker_args(parser)
        ranking_model = NeuralRanker(ranker_args)
        checkpoint = torch.load(args.ranker_checkpoint)
        ranking_model.load_state_dict(checkpoint)
        ranking_model.to(device)
    else:
        logger.info("No ranker is used...")
        ranking_model = None

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
