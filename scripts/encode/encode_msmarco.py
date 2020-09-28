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

from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
from bert_serving.client import BertClient

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def compress(tar_file, members):
    """
    Adds files (`members`) to a tar_file and compress it
    """
    # open file for gzip compressed writing
    tar = tarfile.open(tar_file, mode="w:gz")
    # with progress bar
    # set the progress bar
    progress = tqdm(members)
    for member in progress:
        # add file/folder/link to the tar file (compress)
        tar.add(member)
        # set the progress description of the progress bar
        progress.set_description(f"Compressing {member}")
    # close the file
    tar.close()


def encode_passages(args):
    chunks = []
    base_dir = args.base_dir
    starting_chunk = args.starting_chunk
    end_chunk = args.end_chunk
    limit = args.limit
    step_size = args.step_size
    bert_client = args.bert_client
    zip_chunks = args.zip_chunks
    output = args.out_dir
    print('loaded args in encode_passages..')

    for chunk_id in tqdm(range(starting_chunk, end_chunk, step_size)):

        print('load chunks and set file name for encoded passages..')
        fname = base_dir + 'chunks/' + str(chunk_id) + '_passage_collection_' + str(limit) + '.tsv'  ### id \t passage
        with open(fname, 'r', encoding="utf8") as corpus_in:
            print(f'current chunk opened')

            fname = base_dir + output + str(chunk_id) + '_encoded_passages_' + str(limit) + '.json'

            # save encoded name in list for compressing later
            if fname not in chunks:
                chunks.append(fname)

            # check if file name already exists, skip
            if not os.path.isfile(fname):
                passage_dict = dict()

                passages = []
                pids = []

                # load passages and pids, take time for monitoring
                #start = default_timer()
                for line in tqdm(corpus_in):
                    split = line.strip().split('\t')
                    passages.append(split[1])
                    pids.append(split[0])
                #end = default_timer()
                #length = len(passages) // 2
                #print(f"Time for appending {chunk_id}-th chunk: {end - start}s")
                #logger.info(f"Time for appending {chunk_id}-th chunk: {end - start}s")
                start = default_timer()
                encoded_passages = bert_client.encode(passages).tolist()
                end = default_timer()
                print(f"Time for encoding {chunk_id} current chunk: {end - start}s")
                #logger.info(f"Time for encoding half of {chunk_id} current chunk: {end - start}s")
                #encoded_passages2 = bert_client.encode(passages[length:]).tolist()
                #start = default_timer()
                #print(f"Time for encoding second half of {chunk_id} current chunk: {start - end}s")
                #logger.info(f"Time for encoding second half of {chunk_id} current chunk: {start - end}s")
                #encoded_passages.extend(encoded_passages2)

                # create passage dict to store in json
                j = 0
                for pid in tqdm(pids):
                    passage_dict[pid] = encoded_passages[j]
                    j += 1

                with open(fname, 'w') as fp:
                    json.dump(passage_dict, fp)

                # for monitoring
                #fname = base_dir + 'last_odd_chunk.txt'
                #with open(fname, 'w') as fp:
                #    fp.write(str(chunk_id) + '\n')
            else:
                print(f'chunk id {chunk_id} already encoded...')
                #logger.info(f'chunk id {chunk_id} already encoded...')

            if zip_chunks > 0:
                if chunk_id != 0 and len(chunks) >= zip_chunks:
                    # add current chunks to zip
                    print('compressing current chunks...')
                    #logger.info('compressing current chunks...')
                    tar_name = base_dir + 'tars_odd/' + str(starting_chunk) + '_to_' + str(chunk_id) + '_odd.tar.gz'
                    compress(tar_name, chunks)
                    chunks = []

            corpus_in.close()


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
        encoded_queries = bert_client.encode(queries)
        logger.info('encoding done!')
        if args.output_numpy:
            logger.info('save qids and queries as .npy')
            qids = np.asarray(qids)
            np.save(os.path.join(args.base_dir, 'msmarco_queries.npy'), encoded_queries)
            np.save(os.path.join(args.base_dir, 'msmarco_qids.npy'), qids)
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


def convert_to_numpy(args):
    passages = []
    pids = []
    for chunk_id in tqdm(range(0, args.end_chunk)):
        if chunk_id % 100 == 0:
            logger.info(f'loading chunk {chunk_id}/{args.end_chunk} into list')
        chunk_file = os.path.join(args.base_dir, args.out_dir, str(chunk_id) + '_encoded_passages_' + args.limit + '.json')

        with open(chunk_file, 'r') as f:
            passage_dict = json.load(f)
            for pid in passage_dict:
                passages.append(passage_dict[pid])
                pids.append(pid)
    fname = os.path.join(args.base_dir, 'msmarco_encoded_passages.npy')
    logger.info(f'convert passages to numpy and store into file: {fname}')
    passages = np.asarray(passages)
    np.save(fname, passages)
    fname = os.path.join(args.base_dir, 'pids_msmarco_encoded_passages.npy')
    logger.info(f'convert passages to numpy and store into file: {fname}')
    pids = np.asarray(pids)
    np.save(fname, pids)


if __name__ == '__main__':
    #Arguments = /home/brandt/Multistep_Query_Modelling/scripts/encode/encode_msmarco.py -model_dir /home/brandt/data/
    #models -base_dir /home/brandt/msmarco/ -end_chunk 1114 -limit 150 -max_seq_len 256 -encode_queries 1 -output_numpy 1



    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('-encode_queries', type=bool, default=False,
                        help='encode msmarco queries')
    parser.add_argument('-query_file_name', type=str, default='docleaderboard-queries.tsv',
                        help='name of the file with qid<tab>query')
    parser.add_argument('-encode_chunks', type=bool, default=True,
                        help='encode msmarco chunks')
    parser.add_argument('-output_numpy', type=bool, default=False,
                        help='if true output in npy file, else every chunk in a single json file')
    parser.add_argument('-convert_to_npy', type=bool, default=False,
                        help='convert already encoded chunks to numpy format')
    parser.add_argument('-base_dir', type=str, default=None,
                        help='base directory of the chunked dataset')
    parser.add_argument('-out_dir', type=str, default=None,
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
    parser.add_argument('-zip_chunks', type=int, default=0,
                        help='number of chunks after which they get compressed into .tar.gz use 0 to not zip')
    parser.add_argument('-num_worker', type=int, default=1,
                        help='number of workers used for bert server, should be less or equal to count of gpus')

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
    if args.convert_to_numpy:
        convert_to_numpy(args)
