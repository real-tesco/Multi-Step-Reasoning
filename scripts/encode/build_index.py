#!/usr/bin/python3

#LEGACY

from timeit import default_timer
import io
import os
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

    logger.info('Starting to load encoded numpy file')
    with open(args.passage_file, 'rb') as f:
        data = np.load(f)
    with open(args.indices_file, 'rb') as f:
        indices = np.load(f).astype(int)

    num_elem = len(indices)
    logger.info('Starting to create knn index...')
    p = hnswlib.Index(space=args.similarity, dim=args.dimension)

    p.init_index(max_elements=num_elem, ef_construction=args.ef_construction, M=args.M)  # parameter tuning
    step_size = num_elem // 100
    for i in range(0, 100):
        if i+1 not in range(0, 100):
            p.add_items(data[i * step_size:], indices[i * step_size:])
        else:
            p.add_items(data[i*step_size:(i+1)*step_size], indices[i*step_size:(i+1)*step_size])
        logger.info(f'Indexed {(i+1) * step_size} / {num_elem} passages!')

    logger.info('Finished creating index, starting saving index')
    index_name = args.out_dir + f'msmarco_knn_index_M_{args.M}_efc_{args.ef_construction}.bin'
    p.save_index(index_name)

    if args.test:
        labels, distances = p.knn_query(data, k=1)
        logger.info("Recall for dataset: ", np.mean(labels.reshape(labels.shape[0]) == indices))


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
                    passage = split[1].replace('"', '').replace("\\", "/").strip('\n')
                    if not isinstance(passage, str):
                        logger.info(f"pid {pid} just got skipped with passage:\n {passage}")
                        continue
                    fout.write(f'{{"id": "{str(pid)}", "contents": "{passage}"}}\n')
    logger.info('Conversion done!')


if __name__ == '__main__':
    #start indexing on hadoop:
    # python3 build_index.py -index_type knn -out_dir ./index/M84ef_construction800/ -ef_construction 800 -M 84 -similarity ip
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('-index_type', type=str, default='knn', choices=['knn', 'bm25'],
                        help='type of index to build: choose from knn and bm25')
    parser.add_argument('-embedding_dir', type=str,
                        help='path to encoded passages, should be in chunks as dicts in .json files with pid:passage')
    parser.add_argument('-out_dir', type=str, default='./',
                        help='output directory for the index')
    parser.add_argument('-similarity', type=str, default='ip', choices=['cosine', 'l2', 'ip'],
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
    parser.add_argument('-passage_file', type=str, default='./msmarco_passages.npy',
                        help='path to the passage encoding file')
    parser.add_argument('-indices_file', type=str, default='./msmarco_indices.npy',
                        help='path to the indices for the passages')
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
    elif args.index_type == 'knn':
        create_knn_index(args)
    elif args.index_type == 'bm25':
        pass

