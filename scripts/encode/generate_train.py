#!/usr/bin/python3

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"

import argparse
import csv
import random
import gzip
import os
from collections import defaultdict
import numpy as np
import json
import pyserini
from pyserini.search import SimpleSearcher
import logging

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
            elif not args.random_sample:
                if unjudged_docid in docid2pids:
                    negatives.append(unjudged_docid)
                if len(negatives) < 5:
                    continue

            already_done_a_triple_for_topicid = topicid

            assert topicid in qrel
            # assert unjudged_docid in docoffset

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
                stats['skipped'] += 1
                continue
            assert positive_docid in docid2pids
            positive_pids = docid2pids[positive_docid]

            stats['kept'] += 1

            # generate positive example, best bm25 passage regarding query, from positive judged document
            query_text = args.queries[topicid]

            hits = searcher.search(query_text)
            if len(hits) == 0:
                stats['skipped'] += 1
                logger.info("skipped another one")
                continue
            best_pid = -1
            for i in range(0, len(hits)):
                if hits[i].docid in positive_pids:
                    best_pid = hits[i].docid
                    logger.info("found pid of correct doc writing that to out")
                    break
            if best_pid == -1:
                best_pid = hits[0].docid
                logger.info("not found pid of correct doc, using top rated pid instead ")
                stats['best_pid_not_in_positive_doc'] += 1

            out.write("{}\t{}\t{}\n".format(topicid, best_pid, random.choice(negative_passages)))

            negatives = []
    return stats


def main(args):
    logger.info("Opening docid2pid...")
    with open(args.docid2pid, 'r') as f:
        docid2pid = json.load(f)

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
    args.docid2pid = docid2pid
    logger.info("Opened files")
    logger.info(f"Loading anserini index from path {args.anserini_index}...")
    searcher = SimpleSearcher(args.anserini_index)
    searcher.set_bm25(0.9, 0.4)
    searcher.set_rm3(10, 10, 0.5)
    args.searcher = searcher

    stats = generate_triples(args)

    for key, val in stats.items():
        logger.info(f"{key}\t{val}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('-docid2pid', type=str, default='docid2pids.json',
                        help='the json file with dict for doc id to passage id mapping')
    parser.add_argument('-base_dir', type=str, help='base directory for files')
    parser.add_argument('-qrels_file', type=str, default='msmarco-doctrain-qrels.tsv')
    parser.add_argument('-triples_name', type=str, default='msmarco_train_triples.tsv')
    parser.add_argument('-doc_train_100_file', type=str, default="msmarco-doctrain-top100")
    parser.add_argument('-anserini_index', type=str, default='indexes/msmarco_passaged_150_anserini/')
    parser.add_argument('-query_file', type=str, default='msmarco-doctrain-queries.tsv')
    parser.add_argument('-doc_lookup', type=str, default='msmarco-docs-lookup.tsv')
    parser.add_argument('-random_sample', type='bool', default=True,
                        help='do the negative sampling at random or take top ranked docs by BM25')

    args = parser.parse_args()

    args.docid2pid = os.path.join(args.base_dir, args.docid2pid)
    args.qrels_file = os.path.join(args.base_dir, args.qrels_file)
    args.doc_train_100_file = os.path.join(args.base_dir, args.doc_train_100_file)
    args.triples_name = os.path.join(args.base_dir, args.triples_name)
    args.anserini_index = os.path.join(args.base_dir, args.anserini_index)
    args.query_file = os.path.join(args.base_dir, args.query_file)
    args.doc_lookup = os.path.join(args.base_dir, args.doc_lookup)

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    main(args)
