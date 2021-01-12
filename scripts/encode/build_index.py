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


logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def create_knn_index(args):

    logger.info('Starting to load encoded numpy file')
    max_elements = args.max_elements

    p = hnswlib.Index(space=args.similarity, dim=args.dimension)
    p.init_index(max_elements=max_elements, ef_construction=args.ef_construction, M=args.M)  # parameter tuning
    idx = 0
    docid2indexid = {}

    for i in range(0, args.number_of_doc_files):
        data = np.load(args.passage_file_format.format(i))
        indices = np.load(args.indices_file_format.format(i))
        print(indices.shape)
        current_idxs = np.empty(len(indices))
        for idy, docid in enumerate(indices):
            current_idxs[idy] = idx
            docid2indexid[docid] = idx
            idx += 1
        logger.info('Starting adding current chunk of docs to knn index...')
        p.add_items(data, current_idxs)
        logger.info(f'Indexed {idx} / {max_elements} passages!')

    logger.info(f'Finished creating index added {idx} chunks, starting saving index and docid2indexid file')
    index_name = args.out_dir + f'msmarco_knn_index_M_{args.M}_efc_{args.ef_construction}.bin'
    p.save_index(index_name)
    with open(args.mapping_file, 'w+') as f:
        json.dump(docid2indexid, f)
    logger.info('Finished!')

    if args.test:
        data = np.load(args.passage_file_format.format(0))
        indices = np.load(args.indices_file_format.format(0))
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


def extend_knn_index(args):
    max_elements = args.max_elements
    index_name = args.out_dir + f'msmarco_firstP_and_remainP_512_knn_index_M_{args.M}_efc_{args.ef_construction}.bin'
    logger.info(f'Start loading existing index from {}...')
    p = hnswlib.Index(space=args.similarity, dim=args.dimension)
    p.load_index(index_name, max_elements=max_elements)
    idx = p.get_current_count()
    logger.info(f"currently the index contains {idx} elements")
    with open(args.mapping_file, "r") as f:
        docid2indexid = json.load(f)

    for i in range(0, args.number_of_doc_files):
        data = np.load(args.passage_file_format.format(i))
        indices = np.load(args.indices_file_format.format(i))
        current_idxs = np.empty(len(indices))
        for idy, docid in enumerate(indices):
            current_idxs[idy] = idx
            if docid in docid2indexid:
                docid2indexid[docid] = (docid2indexid[docid], idx)
            else:
                logger.info(f"docid {docid} was not found in mapping file...")
                docid2indexid[docid] = idx
            idx += 1
        logger.info('Starting adding current chunk of docs to knn index...')
        p.add_items(data, current_idxs)
        logger.info(f'Indexed {idx} / {max_elements} passages!')

    logger.info(f'Finished creating index added {idx} chunks, starting saving index and docid2indexid file')
    p.save_index(index_name)
    with open(args.mapping_file, 'w+') as f:
        json.dump(docid2indexid, f)
    logger.info('Finished!')


if __name__ == '__main__':
    #start indexing on hadoop/nvidia:
    # python3 ./scripts/encode/build_index.py -index_type knn -out_dir ./data/indexes/ -ef_construction 100 -M 84 -similarity ip
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('-index_type', type=str, default='knn', choices=['knn', 'bm25'],
                        help='type of index to build: choose from knn and bm25')
    parser.add_argument('-embedding_dir', type=str,
                        help='path to encoded passages, should be in chunks as dicts in .json files with pid:passage')
    parser.add_argument('-out_dir', type=str, default='./data/indexes/',
                        help='output directory for the index')
    parser.add_argument('-similarity', type=str, default='ip', choices=['cosine', 'l2', 'ip'],
                        help='similarity score to use when knn index is chosen')
    parser.add_argument('-dimension', type=int, default=768,
                        help='dimension of the embeddings for knn index')
    parser.add_argument('-test', type='bool', default=False,
                        help='if true testing recall for knn index with querying dataset and receive top 1')
    parser.add_argument('-ef_construction', type=int, default=400,
                        help='hnswlib parameter, the size of the dynamic list for the nearest neighbors, higher ef'
                             ' leads to higher accuracy but slower search/construction time ')
    parser.add_argument('-M', type=int, default=64,
                        help='hnswlib parameter, the number of bi-directional links created for every new element '
                             'during construction. Range: 0-100. For embeddings 48-64 is reasonable')
    parser.add_argument('-convert_tsv_to_json', type='bool', default=False,
                        help='convert chunks in tsv files in folder to .json files for indexing')
    parser.add_argument('-passage_file_format', type=str, default='./data/embeddings/marco_doc_embeddings_{}.npy',
                        help='path to the passage encoding file')
    parser.add_argument('-indices_file_format', type=str,
                        default='./data/embeddings/marco_doc_embeddings_indices_{}.npy',
                        help='path to the indices for the passages')
    parser.add_argument('-number_of_doc_files', type=int, default=13)
    parser.add_argument('-max_elements', type=int, default=3213835)
    parser.add_argument('-mapping_file', type=str, default='./data/indexes/mapping_docid2indexid.json')
    parser.add_argument('-extend_index', type='bool', default=False)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    args = parser.parse_args()

    if args.convert_tsv_to_json:
        convert_tsv_to_json(args)
    elif args.index_type == 'knn':
        if args.extend_index:
            extend_knn_index(args)
        else:
            create_knn_index(args)
    elif args.index_type == 'bm25':
        pass

