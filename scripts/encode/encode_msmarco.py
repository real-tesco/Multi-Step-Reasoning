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
import sklearn.preprocessing as preprocessing

from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
from bert_serving.client import BertClient

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def encode_passages(args):
    chunks = []
    base_dir = args.base_dir
    starting_chunk = args.starting_chunk
    end_chunk = args.end_chunk
    limit = args.limit
    step_size = args.step_size
    bert_client = args.bert_client
    output = args.out_dir
    print('loaded args in encode_passages..')

    for chunk_id in tqdm(range(starting_chunk, end_chunk, step_size)):

        print('load chunks and set file name for encoded passages..')
        fname = base_dir + 'chunks/' + str(chunk_id) + '_passage_collection_' + str(limit) + '.tsv'  ### id \t passage
        with open(fname, 'r', encoding="utf8") as corpus_in:
            print(f'current chunk opened')

            fname = base_dir + output + str(chunk_id) + '_encoded_passages_' + str(limit) + '.json'

            # check if file name already exists, skip
            if not os.path.isfile(fname):
                passage_dict = dict()

                passages = []
                pids = []

                # load passages and pids, take time for monitoring
                for line in tqdm(corpus_in):
                    split = line.strip().split('\t')
                    passages.append(split[1])
                    pids.append(split[0])

                start = default_timer()
                encoded_passages = bert_client.encode(passages).tolist()
                end = default_timer()
                print(f"Time for encoding {chunk_id} current chunk: {end - start}s")
                # create passage dict to store in json
                for idx, pid in enumerate(tqdm(pids)):
                    passage_dict[pid] = encoded_passages[idx]
                with open(fname, 'w') as fp:
                    json.dump(passage_dict, fp)
            else:
                print(f'chunk id {chunk_id} already encoded...')
                #logger.info(f'chunk id {chunk_id} already encoded...')
            corpus_in.close()


def to_numpy(args):
    passages_to_numpy = []
    pids_to_numpy = []
    for chunk_id in tqdm(range(0, args.end_chunk)):
        fname = args.base_dir + args.out_dir + str(chunk_id) + '_encoded_passages_' + str(args.limit) + '.json'
        with open(fname, 'w') as fp:
            passages = json.load(fp)
            for key in passages:
                passages_to_numpy.append(passages[key])
                pids_to_numpy.append(key)

    steps = len(passages_to_numpy) // args.num_pass_files
    for i in tqdm(range(0, args.num_pass_files)):
        if i+1 not in range(0, args.num_pass_files):
            current_passages = np.asarray(passages_to_numpy[i * steps:]).astype(np.float32)
            current_pids = np.asarray(pids_to_numpy[i * steps:]).astype(int)
        else:
            current_passages = np.asarray(passages_to_numpy[i*steps:(i+1)*steps]).astype(np.float32)
            current_pids = np.asarray(pids_to_numpy[i*steps:(i+1)*steps]).astype(int)
        current_passages = preprocessing.normalize(current_passages, norm="l2")
        fname_passages = args.base_dir + args.out_dir + str(i) + '_msmarco_passages.npy'
        fname_pids = args.base_dir + args.out_dir + str(i) + '_msmarco_pids.npy'
        with open(fname_passages, 'wb') as f:
            np.save(f, current_passages)
        with open(fname_pids, 'wb') as f:
            np.save(f, current_pids)


def encode_queries(args):
    query_file_name = os.path.join(args.base_dir, args.query_file_name)

    qids = []
    queries = []
    bert_client = args.bert_client
    with open(query_file_name) as f:
        for line in f:
            split = line.split('\t')
            qids.append(split[0])
            queries.append(split[1])
        logger.info(f'encoding {len(qids)} queries...')
        print(f'encoding {len(qids)} queries...')
        encoded_queries = bert_client.encode(queries)
        encoded_queries = preprocessing.normalize(encoded_queries, norm="l2").astype(np.float32)
        print(f'encoding of {len(qids)} queries done with bert')
        if args.to_numpy:
            logger.info('save qids and queries as .npy')
            print('save qids and queries as .npy')
            qids = np.asarray(qids)
            np.save(os.path.join(args.base_dir, args.name + '.msmarco_queries_normed.npy'), encoded_queries)
            np.save(os.path.join(args.base_dir, args.name + '.msmarco_qids.npy'), qids)
            print('save qids and queries as .npy DONE')
        else:
            logger.info('save qids and queries as python dict in json')
            encoded_queries = encoded_queries.tolist()
            encoded_query_dict = {}
            i = 0
            for qid in qids:
                encoded_query_dict[qid] = encoded_queries[i]
                i += 1
            with open(os.path.join(args.base_dir, 'msmarco_queries.json'), 'w') as fp:
                json.dump(encoded_query_dict, fp)
        logger.info('queries encoded and saved!')
        print('queries encoded and saved!')

if __name__ == '__main__':
    #Arguments = /home/brandt/Multistep_Query_Modelling/scripts/encode/encode_msmarco.py -model_dir /home/brandt/data/
    #models -base_dir /home/brandt/msmarco/ -end_chunk 1114 -limit 150 -max_seq_len 256 -encode_queries 1 -to_numpy 1

    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('-encode_queries', type='bool', default=False,
                        help='encode msmarco queries')
    parser.add_argument('-query_file_name', type=str, default='msmarco-doctrain-queries.tsv',
                        help='name of the file with qid<tab>query')
    parser.add_argument('-encode_chunks', type='bool', default=False,
                        help='encode msmarco chunks')
    parser.add_argument('-base_dir', type=str, default='',
                        help='base directory of the chunked dataset')
    parser.add_argument('-out_dir', type=str, default='',
                        help='output directory, where to store the embeddings')
    parser.add_argument('-model_dir', type=str, default=None,
                        help='model directory of bert model for bert-as-service')
    parser.add_argument('-max_seq_len', type=int, default=256,
                        help='maximum sequence length used by bert')
    parser.add_argument('-starting_chunk', type=int, default=0,
                        help='id of chunk to start with')
    parser.add_argument('-end_chunk', type=int, default=100,
                        help='number of chunks to encode in this run')
    parser.add_argument('-step_size', type=int, default=1,
                        help='step size to increase the starting chunk id, use 2 and respective start to only encode'
                             'even / odd')
    parser.add_argument('-limit', type=int, default=150,
                        help='limit used for paragraph splitting')
    parser.add_argument('-num_worker', type=int, default=1,
                        help='number of workers used for bert server, should be less or equal to count of gpus')
    parser.add_argument('-name', type=str, default='train', help='name of data to load (dev, train, test)')
    parser.add_argument('-to_numpy', type='bool', default=False,
                        help='convert encoded chunks into -num_pass_files npy files')
    parser.add_argument('-num_pass_files', type=int, default=20, help='number of npy files to store training data in')

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    args = parser.parse_args()

    bert_args = get_args_parser().parse_args(['-model_dir', args.model_dir,
                                              '-port', '5555',
                                              '-port_out', '5556',
                                              '-max_seq_len', str(args.max_seq_len),
                                              '-num_worker', str(args.num_worker)])

    print('starting Bert server and waiting 20 seconds to get it started')
    #logger.info('starting Bert server and waiting 20 seconds to get it started')
    server = BertServer(bert_args)
    server.start()

    time.sleep(20.0)

    # need started server
    bc = BertClient()
    args.bert_client = bc

    #logger.info('testing server: embed text: "hello there, let\'s start encoding" ...')
    print('testing server: embed text: "hello there, let\'s start encoding" ...')
    test = args.bert_client.encode(['Hello there, let\'s start encoding'])
    #logger.info(f'encoding successful, first values of embedding = {test[0][:10]}')
    print(f'encoding successful, first values of embedding = {test[0][:10]}')
    if args.encode_chunks:
        encode_passages(args)
    if args.encode_queries:
        encode_queries(args)
    if args.to_numpy:
        to_numpy(args)
