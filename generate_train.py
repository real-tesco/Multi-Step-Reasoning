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
    """Generates triples comprising:
    - Query: The current topicid and query string
    - Pos: One of the positively-judged documents for this query
    - Rnd: Any of the top-100 documents for this query other than Pos
    Since we have the URL, title and body of each document, this gives us ten columns in total:
    topicid, query, posdocid, posurl, postitle, posbody, rnddocid, rndurl, rndtitle, rndbody
    outfile: The filename where the triples are written
    triples_to_generate: How many triples to generate
    """
    searcher = args.searcher
    docid2pids = args.docid2pid
    qrel = args.qrel
    stats = defaultdict(int)
    already_done_a_triple_for_topicid = -1
    negatives = []

    with open(args.doc_train_100_file, 'rt', encoding='utf8') as top100f, \
            open(args.triples_name, 'w', encoding="utf8") as out:
        for idx, line in enumerate(top100f):
            [topicid, _, unjudged_docid, _, _, _] = line.split()

            if already_done_a_triple_for_topicid == topicid:
                continue
            # getting top 5 documents for query
            elif not args.random_sample:
                if unjudged_docid in docid2pids:
                    negatives.append(unjudged_docid)
                if len(negatives) < 5:
                    continue

            already_done_a_triple_for_topicid = topicid
            assert topicid in qrel

            # generate negative example
            negative_passages = []
            if not args.random_sample:
                # 5 examples top rated in doctrain100 but not in qrels
                negatives_unjudged = [value for value in negatives if value not in qrel[topicid]]
                for neg in negatives_unjudged:
                    for pid in docid2pids[neg]:
                        negative_passages.append(pid)
            else:
                docs = random.choices(args.docids, k=5)
                for doc in docs:
                    if doc in docid2pids:
                        for pid in docid2pids[doc]:
                            negative_passages.append(pid)

            # Use topicid to get our positive_docid
            positive_docid = random.choice(qrel[topicid])
            if positive_docid not in docid2pids:
                stats['skipped_positive_not_in_docid2pid'] += 1
                continue

            assert positive_docid in docid2pids
            positive_pids = docid2pids[positive_docid]
            stats['kept'] += 1

            # generate positive example, best bm25 passage regarding query, from positive judged document
            query_text = args.queries[topicid]

            hits = searcher.search(query_text)
            if len(hits) == 0:
                stats['skipped_hits_len_0'] += 1
                continue
            best_pid = -1
            for i in range(0, min(args.topk, len(hits))):
                if hits[i].docid in positive_pids:
                    best_pid = hits[i].docid
                    stats['best_pid_in_bm25'] += 1
                    break
            if best_pid == -1:
                best_pid = hits[0].docid
                stats['best_pid_not_in_positive_doc'] += 1

            out.write("{}\t{}\t{}\n".format(topicid, best_pid, random.choice(negative_passages)))
            stats['total'] += 1
            if idx % 1000 == 0:
                logger.info(f"{idx} / {len(qrel)} examples done!")
            negatives = []
    return stats


def split_training(args):
    # converts the list of triples to triples with encodings saved in numpy chunks
    encoded_passages = np.load(args.passages)
    encoded_queries = np.load(args.queries)
    queries_indices = np.load(args.queries_indices)

    qid2idx = {}
    for idx, qid in enumerate(queries_indices):
        qid2idx[qid] = idx

    triples_with_encodings = []
    with open(args.triples_name, 'r', encoding="utf8") as f:
        for idx, line in enumerate(f):
            split = line.split('\t')
            assert len(split) == 3
            q = encoded_queries[qid2idx[split[0]]]
            p = encoded_passages[int(split[1]) - 1]
            n = encoded_passages[int(split[0]) - 1]
            triples_with_encodings.append((q, p, n))
            if idx > 0 and idx % 1000 == 0:
                logger.info(f"Loaded {idx}/{len(f)} examples into list")
        logger.info("Converting list to npy and save arrays in chunks")
        triples_with_encodings = np.asarray(triples_with_encodings).astype(np.float32)
    chunk_size = len(triples_with_encodings) // args.split_into_numpy
    for i in range(0, args.split_into_numpy):
        if i+1 not in range(0, args.split_into_numpy):
            tmp_triples = triples_with_encodings[i*chunk_size:]
        else:
            tmp_triples = triples_with_encodings[i*chunk_size:(i+1)*chunk_size]
        np.save(os.path.join(args.out_dir, f"train.triples.{i}.npy"), tmp_triples)
        logger.info(f"Saved {i+1}/{args.split_into_numpy}")
    logger.info("Finished..")


def generate_pairs(args):
    device = args.device
    qrel = args.qrel
    docs = args.docids
    queries = args.queries
    index = args.index
    stats = defaultdict(int)

    with open(args.triples_name, 'w', encoding="utf8") as out:
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
    if args.generate_train:
        generate_train(args)

    if args.split_into_numpy > 0:
        split_training(args)


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
    parser.add_argument('-triples_name', type=str, default='msmarco_train_triples.tsv')
    parser.add_argument('-doc_train_100_file', type=str, default="msmarco-doctrain-top100")
    parser.add_argument('-anserini_index', type=str, default=None)#'indexes/msmarco_passaged_150_anserini/')
    parser.add_argument('-query_file', type=str, default='msmarco-doctrain-queries.tsv')
    parser.add_argument('-doc_lookup', type=str, default='msmarco-docs-lookup.tsv')
    parser.add_argument('-generate_train', type='bool', default=True, help='generate training data')
    parser.add_argument('-queries', type=str, default='embeddings/query_embeddings/train.msmarco_queries_normed.npy',
                        help='all encoded queries in npy')
    parser.add_argument('-queries_indices', type=str,
                        default='embeddings/query_embeddings/train.msmarco_qids.npy',
                        help='all encoded queries in npy')
    parser.add_argument('-passages', type=str, default='input/msmarco_passages_normed_f32.npy',
                        help='all encoded passages in npy')
    parser.add_argument('-passages_indices', type=str, default='input/msmarco_indices.npy')
    parser.add_argument('-out_dir', type=str, help='output directory')

    args = parser.parse_args()

    args.docid2pid = os.path.join(args.base_dir, args.docid2pid)
    args.qrels_file = os.path.join(args.base_dir, args.qrels_file)
    args.doc_train_100_file = os.path.join(args.base_dir, args.doc_train_100_file)
    args.triples_name = os.path.join(args.out_dir, args.triples_name)
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
