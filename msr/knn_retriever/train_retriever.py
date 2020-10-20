import os

import argparse
import torch
from msmarco import MSMARCO
from retriever import KnnIndex
import utilities as utils
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch.optim as optim
import numpy as np
import logging
import json
from torch.autograd import Variable
import math


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


logger = logging.getLogger()

global_timer = utils.Timer()
stats = {'timer': global_timer, 'epoch': 0, 'recall': 0.0}


def triplet_loss(dist_positive, dist_negative, margin=0.3):
    #d = torch.nn.PairwiseDistance(p=2)
    distance = dist_positive - dist_negative + margin
    loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
    return loss


def make_dataloader(queries, qids, pid2docid, triples, triple_ids, train_time=False):
    dataset = MSMARCO(queries, qids, pid2docid, triples, triple_ids, train_time=train_time)
    sampler = SequentialSampler(dataset) if not train_time else RandomSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.data_workers,
        collate_fn=utils.batchify(args, train_time=train_time), #TODO: write batch function for dataloader
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
    return qrel


#TODO: look here
def train_binary_classification(args, ret_model, optimizer, train_loader, verified_dev_loader=None):

    #args.train_time = True
    para_loss = utils.AverageMeter()
    ret_model.query_transformer.train()
    ret_model.document_transformer.train()
    for idx, ex in enumerate(train_loader):
        if ex is None:
            continue

        inputs = [e if e is None or type(e) != type(ex[0]) else Variable(e.cuda())
                  for e in ex[:3]]
        ret_input = [*inputs[:]]
        logger.info(f"reformulated input: {len(ret_input)}")
        logger.info(f"q input: {ret_input[0].shape}")
        logger.info(f"p input: {ret_input[1].shape}")
        logger.info(f"n input: {ret_input[2].shape}")
        scores_positive, scores_negative = ret_model.score_documents(*ret_input) #todo: look here

        logger.info(f"positive score: {scores_positive.shape}")
        logger.info(f"positive score: {scores_positive}")
        logger.info(f"negative score: {scores_negative.shape}")
        logger.info(f"negative score: {scores_negative}")

        # Triplet logits loss
        batch_loss = triplet_loss(scores_positive, scores_negative)
        optimizer.zero_grad()
        batch_loss.backward()

        torch.nn.utils.clip_grad_norm(ret_model.get_trainable_parameters(),
                                      2.0)
        optimizer.step()
        para_loss.update(batch_loss.data.item())
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

    #load data from files
    logger.info('loading files and initializing dataloader...')
    logger.info(f'using cuda: {args.cuda}')
    #if args.cuda:
    #    passages = torch.cuda.FloatTensor(np.load(args.passage_file))
    #else:
    #    passages = np.load(args.passage_file)
    # pids = np.load(args.pid_file)

    #triples = np.load(args.triples_file)
    #triple_ids = np.load(args.triple_ids_file)
    queries = np.load(args.query_file)
    qids = np.load(args.qid_file)
    #qrels = load_qrels(args.qrels_file)
    with open(args.pid2docid, 'r') as f:
        pid2docid = json.load(f)

    #training_loader = make_dataloader(queries, qids, pid2docid, triples, triple_ids, train_time=True)

    # initialize Model
    if args.checkpoint:
        pass
    else:
        logger.info('Initializing model from scratch...')
        retriever_model, optimizer = init_from_scratch(args)

    logger.info("Starting training...")
    for epoch in range(0, args.epochs):
        #train for 1 epoch
        #evaluation for this epoch
        #propagate loss back though network

        stats['epoch'] = epoch

        #need to load the training data in chunks since its too big
        for i in range(0, args.num_training_files):
            triples = np.load(os.path.join(args.training_folder, "train.triples_msmarco" + str(i) + ".npy"))
            triple_ids = np.load(os.path.join(args.training_folder, "msmarco_indices_" + str(i) + ".npy"))

            training_loader = make_dataloader(queries, qids, pid2docid, triples, triple_ids, train_time=True)

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
    parser.add_argument('-pid_folder', type=str, default='msmarco_passage_encodings/', help='name of pids file')
    parser.add_argument('-passage_folder', type=str, default='msmarco_passage_encodings/', help='name of folder with msmarco passage embeddings')
    parser.add_argument('-triples_file', type=str, default='train.triples_msmarco.npy', help='name of triples file with training data')
    parser.add_argument('-triple_ids_file', type=str, default='train.triples.idx_msmarco.npy')
    parser.add_argument('-weight_decay', type=float, default=0, help='Weight decay (default 0)')
    parser.add_argument('-learning_rate', type=float, default=0.1, help='Learning rate for SGD (default 0.1)')
    parser.add_argument('-momentum', type=float, default=0, help='Momentum (default 0)')
    parser.add_argument('-checkpoint', type=bool, default=False, help='Wether to use a checkpoint or not')
    parser.add_argument('-model_name', type=str, default='', help='Model name to load from as checkpoint')
    parser.add_argument('-cuda', type=bool, default=torch.cuda.is_available(), help='use cuda and gpu')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size to use')
    parser.add_argument('-data_workers', type=int, default=5, help='number of data workers to use')
    parser.add_argument('-training_folder', type=str, default='train/', help='folder with chunks of training triples')
    parser.add_argument('-num_training_files', type=int, default=10, help='number of chunks of training triples')


    args = parser.parse_args()

    args.index = os.path.join(args.base_dir, args.index)
    args.query_file = os.path.join(args.base_dir, args.query_file)
    args.pid2docid = os.path.join(args.base_dir, args.pid2docid)
    args.qrels_file = os.path.join(args.base_dir, args.qrels_file)
    args.qid_file = os.path.join(args.base_dir, args.qid_file)
    args.passage_folder = os.path.join(args.base_dir, args.passage_folder)
    args.pid_folder = os.path.join(args.base_dir, args.pid_folder)
    args.triples_file = os.path.join(args.base_dir, args.triples_file)
    args.triple_ids_file = os.path.join(args.base_dir, args.triple_ids_file)
    args.training_folder = os.path.join(args.base_dir, args.training_folder)

    args.state_dict = None
    args.train = True

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    main(args)
