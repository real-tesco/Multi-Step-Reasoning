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

    parser.add_argument('-pid2docid', type=str, default='passage_to_doc_id_150.json',
                        help='name of passage to doc file')

    parser.add_argument('-doc_embedding_format', type=str, default='./data/embeddings/marco_doc_embeddings_{}.npy',
                        help='folder with chunks of document embeddings, with format brackets for idx')
    parser.add_argument('-doc_ids_format', type=str, default='./data/embeddings/marco_doc_embeddings_indices_{}.npy',
                        help='folder with chunks of document ids, with format brackets for idx')
    parser.add_argument('-num_doc_files', type=int, default=13, help='number of chunks of training triples')
    parser.add_argument('-query_embedding_format', type=str,
                        default='./data/embeddings/marco_train_query_embeddings_{}.npy',
                        help='folder with chunks of document embeddings, with format brackets for idx')
    parser.add_argument('-query_ids_format', type=str,
                        default='./data/embeddings/marco_train_query_embeddings_indices_{}.npy',
                        help='folder with chunks of document ids, with format brackets for idx')
    parser.add_argument('-num_query_files', type=int, default=1, help='number of chunks of training triples')
    parser.add_argument('-triples', type=str, default='./data/trids_marco-doc-10.tsv',
                        help='number of chunks of training triples')

    parser.add_argument('-dev_file', type=str, default='./data/msmarco-doc.dev.jsonl')
    parser.add_argument('-dev_query_embedding_file', type=str,
                        default='./data/embeddings/marco_dev_query_embeddings_0.npy')
    parser.add_argument('-dev_query_ids_file', type=str,
                        default='./data/embeddings/marco_dev_query_embeddings_indices_0.npy')
    parser.add_argument('-print_every', type=int, default=100)
    parser.add_argument('-eval_every', type=int, default=10000)

    parser.add_argument('-out_dir', type=str, default='', help='directory for output')
    parser.add_argument('-trec_eval', type=str, default='', help='path to the trec eval file')


    # run options
    parser.add_argument('-test', type='bool', default=True, help='test the index for self-recall and query recall')
    parser.add_argument('-train', type='bool', default=True, help='train document transformer')

    parser.add_argument('-qrels', type=str, default='./data/msmarco-docdev-qrels.tsv',
                        help='dev qrels file')
    parser.add_argument('-out_file', type=str, default='./results/ranking_results.tsv',
                        help='result file for the evaluation of the index')

    args = parser.parse_args()

    '''
    args.pid2docid = os.path.join(args.base_dir, args.pid2docid)
    args.qrels_file = os.path.join(args.base_dir, args.qrels_file)
    args.qid_file = os.path.join(args.base_dir, args.qid_file)
    args.training_folder = os.path.join(args.base_dir, args.training_folder)
    args.dev_queries = os.path.join(args.base_dir, args.dev_queries)
    args.dev_qids = os.path.join(args.base_dir, args.dev_qids)
    args.out_file = os.path.join(args.out_dir, args.out_file)
    args.trec_eval = os.path.join(args.base_dir, args.trec_eval)
    args.qrels_dev_file = os.path.join(args.base_dir, args.qrels_dev_file)
    '''

    return args
