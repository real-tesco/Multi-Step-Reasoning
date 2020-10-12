import os

import argparse
import torch
from msmarco import MSMARCO
from retriever import KnnIndex
import msr.knn_retriever.utils as utils
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch.optim as optim
import numpy as np
import logging
import json
from torch.autograd import Variable


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


logger = logging.getLogger()

global_timer = utils.Timer()
stats = {'timer': global_timer, 'epoch': 0, 'recall': 0.0}


def make_dataloader(passages, pids, queries, qids, pid2docid, qrels, triples, train_time=False):
    dataset = MSMARCO(passages, pids, queries, qids, pid2docid, qrels, triples, train_time=train_time)
    sampler = SequentialSampler(dataset) if not train_time else RandomSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.data_workers,
        collate_fn=utils.batchify(args, args.para_mode, train_time=train_time), #TODO: write batch function for dataloader
        pin_memory=True
    )
    return loader


def init_from_scratch(args):
    retriever_model = KnnIndex(args)
    parameters = retriever_model.get_trainable_parameters()

    optimizer = None
    if parameters is not None and len(parameters) > 0:
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(parameters, args.learning_rate,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.optimizer == 'adamax':
            optimizer = optim.Adamax(parameters,
                                     weight_decay=args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % args.optimizer)
    else:
        pass

    return retriever_model, optimizer


def load_qrels(path_to_qrels):
    qrel = {}
    with open(path_to_qrels, 'rt', encoding='utf8') as f:
        for line in f:
            split = line.split()
            assert len(split) == 4
            topicid, _, docid, rel = split
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]


#TODO: look here
def train_binary_classification(args, ret_model, optimizer, train_loader, verified_dev_loader=None):

    args.train_time = True
    para_loss = utils.AverageMeter()
    ret_model.query_transformer.train()
    for idx, ex in enumerate(train_loader):
        if ex is None:
            continue

        inputs = [e if e is None or type(e) != type(ex[0]) else Variable(e.cuda())
                  for e in ex[:]]
        ret_input = [*inputs[:4]]
        scores, _, _ = ret_model.score_documents(*ret_input) #todo: look here
        y_num_occurrences = Variable(ex[-2])
        labels = (y_num_occurrences > 0).float()
        labels = labels.cuda()
        # BCE logits loss
        batch_para_loss = F.binary_cross_entropy_with_logits(scores.squeeze(1), labels)
        optimizer.zero_grad()
        batch_para_loss.backward()

        torch.nn.utils.clip_grad_norm(ret_model.get_trainable_params(),
                                      2.0)
        optimizer.step()
        para_loss.update(batch_para_loss.data.item())
        if math.isnan(para_loss.avg):
            import pdb
            pdb.set_trace()

        if idx % 25 == 0 and idx > 0:
            logger.info('Epoch = {} | iter={}/{} | para loss = {:2.4f}'.format(
                stats['epoch'],
                idx, len(train_loader),
                para_loss.avg))
            para_loss.reset()


def main(args):

    # initialize Model
    if args.checkpoint:
        pass
    else:
        retriever_model, optimizer = init_from_scratch(args)

    #load data from files
    logger.info('loading files...')
    passages = np.load(args.passage_file)
    pids = np.load(args.pid_file)
    queries = np.load(args.query_file)
    qids = np.load(args.qid_file)
    qrels = load_qrels(args.qrels_file)
    with open(args.pid2docid, 'r') as f:
        pid2docid = json.load(f)
    with open(args.triples_file, 'r') as f:
        #key qid -> (key 'pos' -> list of positive , key 'neg' -> list of negatives)
        triples = json.load(f)

    # load training data with data loaders
    training_loader = make_dataloader(passages, pids, queries, qids, pid2docid, qrels, triples, train_time=True)

    logger.info("Starting training")
    for epoch in range(0, args.epochs):
        #train for 1 epoch
        #evaluation for this epoch
        #propagate loss back though network
        stats['epoch'] = epoch
        train_binary_classification(args, retriever_model, optimizer, training_loader, verified_dev_loader=None)

        #TODO:checkpointing
        """logger.info('checkpointing  model at {}'.format(args.model_file))
        ## check pointing##
        save(args, ret_model.model, optimizer, args.model_file + ".ckpt", epoch=stats['epoch'])
        """
        #TODO:eval
        """logger.info("Evaluating on the full dev set....")
        top1 = eval_binary_classification(args, ret_model, all_dev_exs, dev_loader, verified_dev_loader=None)
        if stats['best_acc'] < top1:
            stats['best_acc'] = top1
            logger.info('Best accuracy {}'.format(stats['best_acc']))
            logger.info('Saving model at {}'.format(args.model_file))
            logger.info("Logs saved at {}".format(args.log_file))
            save(args, ret_model.model, optimizer, args.model_file, epoch=stats['epoch'])
        """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument('-similarity', type=str, default='ip', choices=['cosine', 'l2', 'ip'],
                        help='similarity score to use when knn index is chosen')
    parser.add_argument('-dim', type=int, default=768,
                        help='dimension of the embeddings for knn index')
    parser.add_argument('-optimizer', type=str, default='adamax',
                        help='optimizer to use for training [sgd, adamax]')
    parser.add_argument('-epochs', type=int, default=30,
                        help='number of epochs to train the retriever')
    parser.add_argument('-base_dir', type=str, help='base directory of training/evaluation files')
    parser.add_argument('-index', type=str, default='msmarco_knn_index_M_96_efc_300.bin',
                        help='path to the hnswlib vector index')
    parser.add_argument('-query_file', type=str, default='train.msmarco_queries_normed.npy', help='name of query file')
    parser.add_argument('-qid_file', type=str, default='train.msmarco_qids.npy', help='name of qid file')
    parser.add_argument('-qrels_file', type=str, default='qrels.train.tsv', help='name of qrels file')
    parser.add_argument('-pid2docid', type=str, default='passage_to_doc_id_150.json', help='name of passage to doc file')
    parser.add_argument('-pid_file', type=str, default='msmarco_indices.npy', help='name of pids file')
    parser.add_argument('-passage_file', type=str, default='msmarco_passages_normed.npy', help='name of qrels file')
    parser.add_argument('-triples_file', type=str, default='triples.json', help='name of triples file with training data')
    parser.add_argument('-weight_decay', type=float, default=0, help='Weight decay (default 0)')
    parser.add_argument('-learning_rate', type=float, default=0.1, help='Learning rate for SGD (default 0.1)')
    parser.add_argument('-momentum', type=float, default=0, help='Momentum (default 0)')
    args = parser.parse_args()

    args.index = os.path.join(args.base_dir, args.index)
    args.query_file = os.path.join(args.base_dir, args.query_file)
    args.pid2docid = os.path.join(args.base_dir, args.pid2docid)
    args.qrels_file = os.path.join(args.base_dir, args.qrels_file)
    args.qid_file = os.path.join(args.base_dir, args.qid_file)
    args.passage_file = os.path.join(args.base_dir, args.passage_file)
    args.pid_file = os.path.join(args.base_dir, args.pid_file)
    args.triples_file = os.path.join(args.base_dir, args.triples_file)

    main(args)
