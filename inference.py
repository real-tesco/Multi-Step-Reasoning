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
from msr.utils import Timer
import logging


logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def inference(args, knn_index, ranking_model, dev_loader, metric, device):
    rst_dict = {}
    timer = Timer()
    for idx, dev_batch in enumerate(dev_loader):
        if dev_batch is None:
            continue
        query_id = dev_batch['query_id']
        document_labels, document_embeddings, distances, query_embeddings = knn_index.knn_query_inference(
            dev_batch['q_input_ids'].to(device),
            dev_batch['q_input_mask'].to(device),
            dev_batch['q_segment_ids'].to(device),
            k=100)
        if args.full_ranking:
            batch_score = ranking_model.rerank_documents(query_embeddings.to(device), document_embeddings.to(device), device)
        else:
            batch_score = distances
        # TODO Refactor here
        batch_score = batch_score.detach().cpu().tolist()
        for (q_id, d_id, b_s) in zip(query_id, document_labels, batch_score):
            rst_dict[q_id] = [(score, docid) for score, docid in zip(d_id, b_s)]

        if (idx+1) % args.print_every == 0:
            logger.info(f"{idx+1} / {len(dev_loader)}")
    timer.stop()
    msr.utils.save_trec_inference(args.res, rst_dict)
    if args.metric.split('_')[0] == 'mrr':
        mes = metric.get_mrr(args.qrels, args.res, args.metric)
    else:
        mes = metric.get_metric(args.qrels, args.res, args.metric)
    logger.info(f"Evaluation done: {args.metric}={mes}")
    logger.info(f"Time needed for {len(dev_loader) * args.batch_size}: {timer.time()}")
    logger.info(f"Time needed per query: {timer.time() / (len(dev_loader) * args.batch_size)}")


def main():
    # setting args
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('-two_tower_checkpoint', type=str, default='./checkpoints/twotowerbert.bin')
    parser.add_argument('-ranker_checkpoint', type=str, default='./checkpoints/ranker_extra_layer_2500.ckpt')
    parser.add_argument('-dev_data', action=msr.utils.DictOrStr, default='./data/msmarco-dev-queries-inference.jsonl')
    parser.add_argument('-qrels', type=str, default='./data/msmarco-docdev-qrels.tsv')
    parser.add_argument('-res', type=str, default='./results/twotowerbert.trec')
    parser.add_argument('-metric', type=str, default='mrr_cut_100')
    parser.add_argument('-batch_size', type=int, default='32')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-print_every', type=int, default=25)
    parser.add_argument('-train', type='bool', default=False)
    parser.add_argument('-full_ranking', type='bool', default=True)
    args = parser.parse_args()
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

    # set device
    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading models
    #    1. Load Retriever
    logger.info("Loading Retriever...")
    #index_args = get_knn_args(parser)
    two_tower_bert = TwoTowerBert(index_args.pretrain)
    checkpoint = torch.load(args.two_tower_checkpoint)
    two_tower_bert.load_state_dict(checkpoint)
    knn_index = KnnIndex(index_args, two_tower_bert)
    logger.info("Load Index File and set ef")
    knn_index.load_index()
    knn_index.set_ef(index_args.efc)
    knn_index.set_device(device)

    if args.full_ranking:
        #   2. Load Ranker
        logger.info("Loading Ranker...")
        #ranker_args = get_ranker_args(parser)
        ranking_model = NeuralRanker(ranker_args)
        checkpoint = torch.load(args.ranker_checkpoint)
        ranking_model.load_state_dict(checkpoint)
        ranking_model.to(device)
    else:
        logger.info("No ranker is used, only eval retrieval part...")
        ranking_model = None



    #set metric
    metric = msr.metrics.Metric()

    # starting inference
    logger.info("Starting inference...")
    inference(args, knn_index, ranking_model, dev_loader, metric, device)


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
