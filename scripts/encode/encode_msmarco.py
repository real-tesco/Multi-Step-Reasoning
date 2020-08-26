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

from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
from bert_serving.client import BertClient


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
    end_chunk = args.end_chunks
    limit = args.limit
    step_size = args.step_size
    bert_client = args.bert_client
    zip_chunks = args.zip_chunks
    output = args.out_dir

    for chunk_id in tqdm(range(starting_chunk, end_chunk, step_size)):

        # load chunks and set file name for encoded passages
        fname = base_dir + 'chunks/' + str(chunk_id) + '_passage_collection_' + str(limit) + '.tsv'  ### id \t passage
        corpus_in = io.open(fname, 'r', encoding="utf8")
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
            start = default_timer()
            for line in tqdm(corpus_in):
                split = line.strip().split('\t')
                passages.append(split[1])
                pids.append(split[0])
            end = default_timer()
            length = len(passages) // 2
            print(f"Time for appending {chunk_id}-th chunk: {end - start}s")
            #logger.info(f"Time for appending {chunk_id}-th chunk: {end - start}s")
            start = default_timer()
            encoded_passages = bert_client.encode(passages[:length]).tolist()
            end = default_timer()
            print(f"Time for encoding half of {chunk_id} current chunk: {end - start}s")
            #logger.info(f"Time for encoding half of {chunk_id} current chunk: {end - start}s")
            encoded_passages2 = bert_client.encode(passages[length:]).tolist()
            start = default_timer()
            print(f"Time for encoding second half of {chunk_id} current chunk: {start - end}s")
            #logger.info(f"Time for encoding second half of {chunk_id} current chunk: {start - end}s")
            encoded_passages.extend(encoded_passages2)

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


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('-base_dir', type=str, default=None,
                        help='base directory of the chunked dataset')
    parser.add_argument('-out_dir', type=str, default=None,
                        help='output directory, where to store the embeddings')
    parser.add_argument('-model_dir', type=str, default=None,
                        help='model directory of bert model for bert-as-service')
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


    args = parser.parse_args()

    bert_args = get_args_parser().parse_args(['-model_dir', args.model_dir,
                                              '-port', '5555',
                                              '-port_out', '5556',
                                              '-max_seq_len', '170',
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

    encode_passages(args)
