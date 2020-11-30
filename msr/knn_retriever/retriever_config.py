import argparse
import os
import logging
import torch
import msr

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # retriever model options
    parser.add_argument('-dim_input', type=int, default=768, help="dimension of the embeddings for knn index")
    parser.add_argument('-dim_hidden', type=int, default=768,
                        help='hidden dimension of paragraphs, used for knn index.')
    parser.add_argument('-optimizer', type=str, default='adamax',
                        help='optimizer to use for training [sgd, adamax]')
    parser.add_argument('-pretrain', type=str, default='bert-base-uncased', help='checkpoint file to load checkpoint')
    parser.add_argument('-checkpoint', type='bool', default=False, help='Wether to use a checkpoint or not')
    parser.add_argument('-model_name', type=str, default='knn_index', help='Model name to load from/save as checkpoint')
    parser.add_argument('-max_query_len', type=int, default=64)
    parser.add_argument('-max_doc_len', type=int, default=512)

    # training options
    parser.add_argument('-epochs', type=int, default=3,
                        help='number of epochs to train the retriever')
    parser.add_argument('-weight_decay', type=float, default=0, help='Weight decay (default 0)')
    parser.add_argument('-learning_rate', type=float, default=0.1, help='Learning rate for SGD (default 0.1)')
    parser.add_argument('-momentum', type=float, default=0, help='Momentum (default 0)')
    parser.add_argument('-cuda', type=bool, default=torch.cuda.is_available(), help='use cuda and gpu')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size to use')
    parser.add_argument('-data_workers', type=int, default=5, help='number of data workers to use')
    parser.add_argument('-margin', type=float, default=0.5, help='value for the margin to use in triplet loss')

    # file options
    parser.add_argument('-index_file', type=str, default='./data/indexes/msmarco_knn_index_M_84_efc_500.bin')
    parser.add_argument('-index_mapping', type=str, default='./data/indexes/mapping_docid2indexid.json')
    parser.add_argument('-pid2docid', type=str, default='passage_to_doc_id_150.json',
                        help='name of passage to doc file')
    parser.add_argument('-pid_folder', type=str, default='msmarco_passage_encodings/', help='name of pids file')
    parser.add_argument('-passage_folder', type=str, default='msmarco_passage_encodings/',
                        help='name of folder with msmarco passage embeddings')
    parser.add_argument('-num_passage_files', type=int, default=20,
                        help='number of passage files and indices in folder')
    parser.add_argument('-num_training_files', type=int, default=10, help='number of chunks of training triples')

    # Index options
    parser.add_argument('-efc', type=int, default=500, help='efc parameter of hnswlib to create knn index')
    parser.add_argument('-M', type=int, default=84, help='M parameter of hnswlib to create knn index')
    parser.add_argument('-max_elems', type=int, default=22292343, help='maximum number of elements in index')
    parser.add_argument('-similarity', type=str, default='ip', choices=['cosine', 'l2', 'ip'],
                        help='similarity score to use when knn index is chosen')
    parser.add_argument('-start_chunk', type=int, default=0,
                        help='chunk to start indexing with, useful if index construction failed midway')
    # run options
    parser.add_argument('-test', type='bool', default=True, help='test the index for self-recall and query recall')
    parser.add_argument('-train', type='bool', default=True, help='train document transformer')
    parser.add_argument('-train_data', action=msr.utils.DictOrStr, default='./data/train_toy.jsonl')
    parser.add_argument('-index', type='bool', default=True, help='create knn index with transformed passages')

    parser.add_argument('-dev_queries', type=str, default='./data/embedings/marco_dev_query_embeddings_0.npy',
                        help='dev query file .npy')
    parser.add_argument('-dev_qids', type=str, default='./data/embeddings/marco_dev_query_embeddings_indices_0.npy', help='dev qids file .npy')
    parser.add_argument('-out_file', type=str, default='./results/results.tsv',
                        help='result file for the evaluation of the index')
    parser.add_argument('-dev_data', type=str, default='', help='WORKAROUND BECAUSE PARSER ERROR')
    parser.add_argument('-res', type=str, default='', help='WORKAROUND BECAUSE PARSER ERROR')
    parser.add_argument('-max_input', type=int, default='', help='WORKAROUND BECAUSE PARSER ERROR')
    args = parser.parse_args()

    return args
