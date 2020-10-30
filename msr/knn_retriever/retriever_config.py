import argparse
import os
import logging
import torch

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # retriever model options
    parser.add_argument('-dim_input', type=int, default=768, help="dimension of the embeddings for knn index")
    parser.add_argument('-dim_hidden', type=int, default=300,
                        help='hidden dimension of paragraphs, used for knn index.')
    parser.add_argument('-optimizer', type=str, default='adamax',
                        help='optimizer to use for training [sgd, adamax]')
    parser.add_argument('-pretrained', type=str, default='knn_index.max', help='checkpoint file to load checkpoint')
    parser.add_argument('-checkpoint', type='bool', default=False, help='Wether to use a checkpoint or not')
    parser.add_argument('-model_name', type=str, default='knn_index', help='Model name to load from/save as checkpoint')

    # training options
    parser.add_argument('-epochs', type=int, default=30,
                        help='number of epochs to train the retriever')
    parser.add_argument('-weight_decay', type=float, default=0, help='Weight decay (default 0)')
    parser.add_argument('-learning_rate', type=float, default=0.1, help='Learning rate for SGD (default 0.1)')
    parser.add_argument('-momentum', type=float, default=0, help='Momentum (default 0)')
    parser.add_argument('-cuda', type=bool, default=torch.cuda.is_available(), help='use cuda and gpu')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size to use')
    parser.add_argument('-data_workers', type=int, default=5, help='number of data workers to use')

    # file options
    parser.add_argument('-base_dir', type=str, help='base directory of training/evaluation files')
    parser.add_argument('-query_file', type=str, default='train.msmarco_queries_normed.npy', help='name of query file')
    parser.add_argument('-qid_file', type=str, default='train.msmarco_qids.npy', help='name of qid file')
    parser.add_argument('-qrels_file', type=str, default='qrels.train.tsv', help='name of qrels file')
    parser.add_argument('-pid2docid', type=str, default='passage_to_doc_id_150.json',
                        help='name of passage to doc file')
    parser.add_argument('-pid_folder', type=str, default='msmarco_passage_encodings/', help='name of pids file')
    parser.add_argument('-passage_folder', type=str, default='msmarco_passage_encodings/',
                        help='name of folder with msmarco passage embeddings')
    parser.add_argument('-num_passage_files', type=int, default=20,
                        help='number of passage files and indices in folder')
    parser.add_argument('-triples_file', type=str, default='train.triples_msmarco.npy',
                        help='name of triples file with training data')
    parser.add_argument('-triple_ids_file', type=str, default='train.triples.idx_msmarco.npy')
    parser.add_argument('-training_folder', type=str, default='train/', help='folder with chunks of training triples')
    parser.add_argument('-num_training_files', type=int, default=10, help='number of chunks of training triples')
    parser.add_argument('-model_file', type=str, default='knn_index', help='Model file to store checkpoint')
    parser.add_argument('-out_dir', type=str, default='', help='Model file to store checkpoint')
    parser.add_argument('-trec_eval', type=str, default='', help='path to the trec eval file')

    # Index options
    parser.add_argument('-efc', type=int, default=300, help='efc parameter of hnswlib to create knn index')
    parser.add_argument('-M', type=int, default=96, help='M parameter of hnswlib to create knn index')
    parser.add_argument('-max_elems', type=int, default=22292343, help='maximum number of elements in index')
    parser.add_argument('-similarity', type=str, default='ip', choices=['cosine', 'l2', 'ip'],
                        help='similarity score to use when knn index is chosen')
    parser.add_argument('-hnsw_index', type=str, default='msmarco_knn_M_96_efc_300.bin',
                        help='create knn index with transformed passages')
    parser.add_argument('-start_chunk', type=int,
                        help='chunk to start indexing with, useful if index construction failed midway')
    # run options
    parser.add_argument('-test', type='bool', default=True, help='test the index for self-recall and query recall')
    parser.add_argument('-train', type='bool', default=True, help='train document transformer')
    parser.add_argument('-index', type='bool', default=True, help='create knn index with transformed passages')

    parser.add_argument('-dev_queries', type=str, default='dev.msmarco_queries_normedf32.npy',
                        help='dev query file .npy')
    parser.add_argument('-dev_qids', type=str, default='dev.msmarco_qids.npy', help='dev qids file .npy')
    parser.add_argument('-out_file', type=str, default='results.tsv',
                        help='result file for the evaluation of the index')

    args = parser.parse_args()

    args.query_file = os.path.join(args.base_dir, args.query_file)
    args.pid2docid = os.path.join(args.base_dir, args.pid2docid)
    args.qrels_file = os.path.join(args.base_dir, args.qrels_file)
    args.qid_file = os.path.join(args.base_dir, args.qid_file)
    args.passage_folder = os.path.join(args.base_dir, args.passage_folder)
    args.pid_folder = os.path.join(args.base_dir, args.pid_folder)
    args.triples_file = os.path.join(args.base_dir, args.triples_file)
    args.triple_ids_file = os.path.join(args.base_dir, args.triple_ids_file)
    args.training_folder = os.path.join(args.base_dir, args.training_folder)
    args.model_file = os.path.join(args.out_dir, args.model_file)
    args.dev_queries = os.path.join(args.base_dir, args.dev_queries)
    args.dev_qids = os.path.join(args.base_dir, args.dev_qids)
    args.out_file = os.path.join(args.out_dir, args.out_file)
    args.hnsw_index = os.path.join(args.out_dir, args.hnsw_index)
    args.trec_eval = os.path.join(args.base_dir, args.trec_eval, "trec_eval")
    args.state_dict = None
    return args
