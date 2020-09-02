#!/usr/bin/python3

from timeit import default_timer
import io
import os.path
import json
import tarfile
from tqdm import tqdm  # pip3 install tqdm
import logging
import argparse
import time
import sys
import numpy as np
import hnswlib
import pyserini

#import os
#os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

#from jnius import autoclass


logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def create_knn_index(args):
    number_of_chunks = len([name for name in os.listdir(args.embedding_dir)])
    pids = []
    passages = []

    logger.info('Starting to load encoded passage chunks into memory')
    for chunk_id in tqdm(range(0, number_of_chunks)):
        fin = args.embedding_dir + str(chunk_id) + '_encoded_passages_150.json'
        with open(fin, 'r') as f:
            chunk = json.load(f)
            for pid in chunk:
                pids.append(pid)
                passages.append(chunk[pid])
    num_elem = len(pids)
    pids = np.asarray(pids)
    passages = np.asarray(passages)
    logger.info('Starting to create knn index...')
    p = hnswlib.Index(space=args.similarity, dim=args.dimension)

    p.init_index(max_elements=num_elem, ef_construction=400, M=64)  # parameter tuning

    p.add_items(passages, pids)
    logger.info('Finished creating index, starting saving index')
    p.save_index(args.out_dir)

    if args.test:
        labels, distances = p.knn_query(passages, k=1)
        logger.info("Recall for dataset: ", np.mean(labels.reshape(-1) == pids))


def convert_tsv_to_json(args):
    number_of_chunks = len([name for name in os.listdir(args.embedding_dir)])

    logger.info('Starting loading passage chunks and writing to jsonl...')
    fout = args.out_dir + 'full_msmarco_passage_collection_150_pyseriniformat.jsonl'
    current_dict = {}
    with open(fout, 'w', encoding='utf8') as fout:
        for chunk_id in tqdm(range(0, number_of_chunks)):
            fin = args.embedding_dir + str(chunk_id) + '_passage_collection_150.tsv'
            with open(fin, 'r') as f:
                j = 1
                for line in f:
                    #hotfix

                    if chunk_id == 0 and 1308 <= j <= 1310:
                        j += 1
                        continue
                    j += 1
                    #hotfix end
                    split = line.split('\t')
                    pid = split[0]
                    passage = split[1].replace('"', '').strip('\n')
                    if not isinstance(passage, str):
                        logger.info(f"pid {pid} just got skipped with passage:\n {passage}")
                        continue
                    fout.write(f'{{"id": "{str(pid)}", "contents": "{passage}"}}\n')
    logger.info('Conversion done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('-index_type', type=str, default='knn', choices=['knn', 'bm25'],
                        help='type of index to build: choose from knn and bm25')
    parser.add_argument('-embedding_dir', type=str,
                        help='path to encoded passages, should be in chunks as dicts in .json files with pid:passage')
    parser.add_argument('-out_dir', type=str, default='./',
                        help='output directory for the index')
    parser.add_argument('-similarity', type=str, default='cosine', choices=['cosine', 'l2', 'ip'],
                        help='similarity score to use when knn index is chosen')
    parser.add_argument('-dimension', type=int, default=768,
                        help='dimension of the embeddings for knn index')
    parser.add_argument('-test', type=bool, default=False,
                        help='if true testing recall for knn index with querying dataset and receive top 1')
    parser.add_argument('-ef_construction', type=int, default=400,
                        help='hnswlib parameter, the size of the dynamic list for the nearest neighbors, higher ef'
                             ' leads to higher accuracy but slower search/construction time ')
    parser.add_argument('-M', type=int, default=64,
                        help='hnswlib parameter, the number of bi-directional links created for every new element '
                             'during construction. Range: 0-100. For embeddings 48-64 is reasonable')
    parser.add_argument('-convert_tsv_to_json', type=bool, default=False,
                        help='convert chunks in tsv files in folder to .json files for indexing')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    args = parser.parse_args()

    logger.info('testing pyjnius')
    #JString = autoclass('java.lang.String')
    #test = JString('Hello world')
    #logger.info(test)
    if args.convert_tsv_to_json:
        convert_tsv_to_json(args)
    if args.index_type == 'knn':
        pass
        #create_knn_index(args)
    elif args.index_type == 'bm25':
        pass

