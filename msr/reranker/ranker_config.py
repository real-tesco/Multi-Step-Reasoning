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

    # ranker model options
    parser.add_argument('-ranker_input', type=int, default=768*2,
                        help="dimension of the input to the ranker, should be twice of the embedding dim")
    parser.add_argument('-ranker_hidden', type=int, default=768,
                        help='hidden dimension of ranker')
    parser.add_argument('-optimizer', type=str, default='adamax',
                        help='optimizer to use for training [sgd, adamax]')
    parser.add_argument('-pretrained', type=str, default='ranker.ckpt', help='checkpoint file to load checkpoint')
    parser.add_argument('-checkpoint', type='bool', default=False, help='Whether to use a checkpoint or not')
    parser.add_argument('-model_name', type=str, default='ranker', help='Model name to load from/save as checkpoint')

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

    parser.add_argument('-training_folder', type=str, default='train/',
                        help='folder with chunks of training triples')
    parser.add_argument('-num_training_files', type=int, default=10, help='number of chunks of training triples')

    parser.add_argument('-out_dir', type=str, default='', help='directory for output')
    parser.add_argument('-trec_eval', type=str, default='', help='path to the trec eval file')


    # run options
    parser.add_argument('-test', type='bool', default=True, help='test the index for self-recall and query recall')
    parser.add_argument('-train', type='bool', default=True, help='train document transformer')

    parser.add_argument('-dev_queries', type=str, default='dev.msmarco_queries_normedf32.npy',
                        help='dev query file .npy')
    parser.add_argument('-dev_qids', type=str, default='dev.msmarco_qids.npy', help='dev qids file .npy')
    parser.add_argument('-out_file', type=str, default='ranking_results.tsv',
                        help='result file for the evaluation of the index')

    args = parser.parse_args()

    args.query_file = os.path.join(args.base_dir, args.query_file)
    args.pid2docid = os.path.join(args.base_dir, args.pid2docid)
    args.qrels_file = os.path.join(args.base_dir, args.qrels_file)
    args.qid_file = os.path.join(args.base_dir, args.qid_file)
    args.training_folder = os.path.join(args.base_dir, args.training_folder)
    args.dev_queries = os.path.join(args.base_dir, args.dev_queries)
    args.dev_qids = os.path.join(args.base_dir, args.dev_qids)
    args.out_file = os.path.join(args.out_dir, args.out_file)
    args.trec_eval = os.path.join(args.base_dir, args.trec_eval)

    return args
