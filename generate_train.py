#!/usr/bin/python3

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"
import torch
import argparse
import random
import os
import logging
from collections import defaultdict

from tqdm import tqdm
import numpy as np
from pyserini.search import SimpleSearcher

from msr.knn_retriever.retriever import KnnIndex
from msr.knn_retriever.two_tower_bert import TwoTowerBert

logger = logging.getLogger()

# train triples form: Query_id \t positive_id \t negative_id


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def generate_triples(args):
    qrel = args.qrel
    stats = defaultdict(int)
    top100_not_in_qrels = args.top100_not_in_qrels
    negatives = []

    with open(args.out_file, 'w', encoding="utf8") as out:
        for idx, topicid in enumerate(qrel):
            positive_docid = random.choice(qrel[topicid])

            if topicid not in top100_not_in_qrels:
                stats["skipped_topicid_not_in_top100"] += 1
                continue

            # getting top documents but negative for query
            if not args.random_sample:
                negatives = random.choices(top100_not_in_qrels[topicid], k=args.negative_samples)
            else:
                docs = random.choices(args.docids, k=args.negative_samples)
                for doc in docs:
                    if doc not in qrel[topicid]:
                        negatives.append(doc)
                    else:
                        stats['skipped_random_because_relevant'] += 1

            for neg in negatives:
                out.write("{}\t{}\t{}\n".format(topicid, positive_docid, neg))
                stats['total'] += 1
            if idx % 1000 == 0:
                logger.info(f"{idx} / {len(qrel)} queries done!")
            negatives = []
    return stats


def generate_pairs(args):
    device = args.device
    qrel = args.qrel
    docs = args.docids
    queries = args.queries
    index = args.index
    stats = defaultdict(int)

    with open(args.out_file, 'w', encoding="utf8") as out:
        for idx, topicid in tqdm(enumerate(qrel)):
            out.write("{} {} {}\n".format(topicid, random.choice(qrel[topicid]), 1))
            stats["kept"] += 1

            for i in range(0, args.negative_samples):
                # random choice
                negative = random.choice(docs)
                if negative in qrel[topicid]:
                    stats["skipped_rnd_neg"] += 1
                    continue
                out.write("{} {} {}\n".format(topicid, negative, 0))
                stats["kept"] += 1

            # negative sampling of top bm25_top_k docs
            if args.use_top_bm25_samples:
                if topicid not in args.top100_not_in_qrels:
                    stats["skipped_not_in_top100"] += 1
                    continue
                negatives = args.top100_not_in_qrels[topicid][:args.topk]
                for i in range(0, args.negative_samples):
                    choice = negatives.pop(random.randrange(0, len(negatives)))
                    out.write("{} {} {}\n".format(topicid, choice, 0))
                    stats["kept"] += 1

            if args.use_knn_index_generation:

                labels, _, _, _ = index.knn_query_text(query_text=queries[topicid], device=device)
                labels = labels[0]
                negatives = []
                search = args.topk
                while len(negatives) < args.topk:
                    negatives = [labels[i] for i in range(search) if labels[i] not in qrel[topicid]]
                    search += 1
                for i in range(0, args.negative_samples):
                    choice = negatives.pop(random.randrange(0, len(negatives)))
                    if choice not in qrel[topicid]:
                        out.write("{} {} {}\n".format(topicid, choice, 0))
                        stats["kept"] += 1

    return stats


def generate_train(args):

    # For each topicid, the list of positive docids is qrel[topicid]
    logger.info("Loading qrel file...")
    qrel = {}
    with open(args.qrels_file, 'rt', encoding='utf8') as f:
        for line in f:
            split = line.split()
            assert len(split) == 4

            topicid, _, docid, rel = split
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
            if len(qrel[topicid]) > 1:
                logger.info("More than 1 docid for this query")

    logger.info("Loading query file...")
    queries = {}
    with open(args.query_file, "r") as f:
        for line in f:
            split = line.split('\t')
            queries[split[0]] = split[1]

    logger.info("Loading doc lookup...")
    docids = []
    with open(args.doc_lookup, "r") as f:
        for line in f:
            split = line.split('\t')
            docids.append(split[0])
    args.docids = docids
    args.queries = queries
    args.qrel = qrel
    # args.docid2pid = docid2pid
    args.searcher = None
    if args.anserini_index is not None:
        logger.info("Opened files")
        logger.info(f"Loading anserini index from path {args.anserini_index}...")
        searcher = SimpleSearcher(args.anserini_index)
        searcher.set_bm25(3.44, 0.87)
        searcher.set_rm3(10, 10, 0.5)
        args.searcher = searcher

    if args.use_top_bm25_samples:
        logger.info("loading top 100 not in qrels per query ")
        top100_not_in_qrels = {}
        with open(args.doc_train_100_file, 'rt', encoding='utf8') as top100f:
            for line in top100f:
                [topicid, _, unjudged_docid, _, _, _] = line.split()
                if unjudged_docid not in qrel[topicid]:
                    if topicid in top100_not_in_qrels:
                        top100_not_in_qrels[topicid].append(unjudged_docid)
                    else:
                        top100_not_in_qrels[topicid] = [unjudged_docid]
        args.top100_not_in_qrels = top100_not_in_qrels

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 0:
        logger.info("using cuda if torch model is used")

    args.index = None
    if args.use_knn_index_generation:
        logger.info(f"Loading Two Tower from {args.use_knn_index_generation}")
        args.index_file = args.use_knn_index_generation
        two_tower_bert = TwoTowerBert(args.pretrain)
        checkpoint = torch.load(args.two_tower_checkpoint)
        two_tower_bert.load_state_dict(checkpoint)
        knn_index = KnnIndex(args, two_tower_bert)
        logger.info("Load Index File and set ef")
        knn_index.load_index()
        knn_index.set_ef(args.efc)
        knn_index.set_device(args.device)

        args.index = knn_index

    if args.pairs:
        stats = generate_pairs(args)
    else:
        stats = generate_triples(args)

    for key, val in stats.items():
        logger.info(f"{key}: \t{val}")


def main(args):
    generate_train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # run options
    parser.add_argument('-random_sample', type='bool', default=True,
                        help='do the negative sampling at random or take top ranked docs by BM25')
    parser.add_argument('-split_into_numpy', type=int, default=0, help='split generated training data into npy')
    parser.add_argument('-negative_samples', type=int, default=2, help='how many negative examples per type')
    parser.add_argument('-pairs', type='bool', default=True, help='create pairs or triples')
    parser.add_argument('-topk', type=int, default=3, help='check if correct passage is under top k of bm25, '
                                                           'else take first passage if passages used')
    parser.add_argument('-use_top_bm25_samples', type='bool', default=True, help='also sample from args.bm25_top_k '
                                                                                 'best docs per query')
    parser.add_argument('-use_knn_index_generation', type=str, default=None, help='use hnswlib index to choose hard'
                                                                                  'examples')

    # hnswlib index options
    parser.add_argument('-index_mapping', type=str, default='indexes/mapping_docid2indexid.json')
    parser.add_argument('-similarity', type=str, default='ip')
    parser.add_argument('-dim_hidden', type=int, default=768)
    parser.add_argument('-efc', type=int, default=100)
    parser.add_argument('-pretrain', type=str, default='bert-base-uncased')
    parser.add_argument('-two_tower_checkpoint', type=str)
    parser.add_argument('-max_doc_len', type=int, default=64, help='64 if queries are used else max doc len eg 512')
    parser.add_argument('-max_query_len', type=int, default=64, help='64 if queries are used else max doc len eg 512')

    #data settings
    parser.add_argument('-docid2pid', type=str, default='docid2pids.json',
                        help='the json file with dict for doc id to passage id mapping')
    parser.add_argument('-base_dir', type=str, help='base directory for files')
    parser.add_argument('-qrels_file', type=str, default='msmarco-doctrain-qrels.tsv')
    parser.add_argument('-out_file', type=str, default='msmarco_train_triples.tsv')
    parser.add_argument('-doc_train_100_file', type=str, default="msmarco-doctrain-top100")
    parser.add_argument('-query_file', type=str, default='msmarco-doctrain-queries.tsv')
    parser.add_argument('-doc_lookup', type=str, default='msmarco-docs-lookup.tsv')

    # for ranker
    parser.add_argument('-anserini_index', type=str, default=None)#'indexes/msmarco_passaged_150_anserini/')
    parser.add_argument('-queries', type=str, default='embeddings/query_embeddings/train.msmarco_queries_normed.npy',
                        help='all encoded queries in npy')
    parser.add_argument('-queries_indices', type=str,
                        default='embeddings/query_embeddings/train.msmarco_qids.npy',
                        help='all encoded queries in npy')
    parser.add_argument('-passages', type=str, default='input/msmarco_passages_normed_f32.npy',
                        help='all encoded passages in npy')
    parser.add_argument('-passages_indices', type=str, default='input/msmarco_indices.npy')

    args = parser.parse_args()

    args.docid2pid = os.path.join(args.base_dir, args.docid2pid)
    args.qrels_file = os.path.join(args.base_dir, args.qrels_file)
    args.doc_train_100_file = os.path.join(args.base_dir, args.doc_train_100_file)
    args.out_file = os.path.join(args.base_dir, args.out_file)
    args.query_file = os.path.join(args.base_dir, args.query_file)
    args.doc_lookup = os.path.join(args.base_dir, args.doc_lookup)

    args.passages = os.path.join(args.base_dir, args.passages)
    args.passages_indices = os.path.join(args.base_dir, args.passages_indices)
    args.queries_indices = os.path.join(args.base_dir, args.queries_indices)
    args.queries = os.path.join(args.base_dir, args.queries)
    if args.anserini_index is not None:
        args.anserini_index = os.path.join(args.base_dir, args.anserini_index)
    if args.use_knn_index_generation is not None:
        args.use_knn_index_generation = os.path.join(args.base_dir, args.use_knn_index_generation)
        args.index_mapping = os.path.join(args.base_dir, args.index_mapping)

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    main(args)
