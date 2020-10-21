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


def make_dataloader(pid2docid, triples, triple_ids, train_time=False):
    dataset = MSMARCO(pid2docid, triples, triple_ids, train_time=train_time)
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


def save(args, model, optimizer, filename, epoch=None):

    params = {'state_dict': {
        'd_transformer': model.document_transformer.state_dict(),
        'q_transformer': model.query_transformer.state_dict(),
        'optimizer': optimizer.state_dict()
    }, 'config': vars(args)}
    if epoch:
        params['epoch'] = epoch
    try:
        torch.save(params, filename)
    except BaseException:
        logger.warn('[ WARN: Saving failed... continuing anyway. ]')


def init_from_checkpoint(args):

    logger.info('Loading model from saved checkpoint {}'.format(args.pretrained))
    checkpoint = torch.load(args.pretrained)
    #args = checkpoint['args']
    args.state_dict = checkpoint['state_dict']
    ret = KnnIndex(args)

    optimizer = None
    parameters = ret.get_trainable_parameters()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(parameters, args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(parameters,
                                 weight_decay=args.weight_decay)
    else:
        raise RuntimeError('Unsupported optimizer: %s' % args.optimizer)
    optimizer.load_state_dict(checkpoint['state_dict']['optimizer'])
    logger.info('Model loaded...')

    return ret, optimizer


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
        #logger.info(f"reformulated input: {ret_input}")
        scores_positive, scores_negative = ret_model.score_documents(*ret_input) #todo: look here

        #logger.info(f"positive score: {scores_positive.shape}")
        #logger.info(f"positive score: {scores_positive}")

        # Triplet loss
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
                idx + stats['chunk']*len(train_loader), len(train_loader)*args.num_training_files,
                para_loss.avg))
            para_loss.reset()


#TODO: implement eval, dev data already on server
def eval_binary_classification(args, ret_model, corpus, dev_loader, verified_dev_loader=None, save_scores = True):
    total_exs = 0
    args.train_time = False
    ret_model.document_transformer.eval()
    ret_model.query_transformer.eval()
    accuracy = 0.0
    for idx, ex in enumerate(dev_loader):
        if ex is None:
            raise BrokenPipeError

        inputs = [e if e is None or type(e) != type(ex[0]) else Variable(e.cuda())
                  for e in ex[:]]
        ret_input = [*inputs[:2]]
        total_exs += ex[0].size(0)

        scores, _, _ = ret_model.score_documents(*ret_input)

        scores = F.sigmoid(scores)
        y_num_occurrences = Variable(ex[-2])
        labels = (y_num_occurrences > 0).float()
        labels = labels.data.numpy()
        scores = scores.cpu().data.numpy()
        scores = scores.reshape((-1))
        if save_scores:
            for i, pid in enumerate(ex[-1]):
                corpus.paragraphs[pid].model_score = scores[i]

        scores = scores > 0.5
        a = scores == labels
        accuracy += a.sum()

    logger.info('Eval accuracy = {} '.format(accuracy/total_exs))
    top1 = get_topk(corpus)
    return top1


def main(args):

    #load data from files
    logger.info('Starting...')
    logger.info(f'using cuda: {args.cuda}')

    with open(args.pid2docid, 'r') as f:
        pid2docid = json.load(f)

    # initialize Model
    if args.checkpoint:
        logger.info('Initializing model from checkpoint...')
        retriever_model, optimizer = init_from_checkpoint(args)
    else:
        logger.info('Initializing model from scratch...')
        retriever_model, optimizer = init_from_scratch(args)

    logger.info("Starting training...")
    for epoch in range(0, args.epochs):
        stats['epoch'] = epoch
        #need to load the training data in chunks since its too big
        for i in range(0, 2):#args.num_training_files):
            logger
            triples = np.load(os.path.join(args.training_folder, "train.triples_msmarco" + str(i) + ".npy"))
            triple_ids = np.load(os.path.join(args.training_folder, "msmarco_indices_" + str(i) + ".npy"))

            stats['chunk'] = i
            training_loader = make_dataloader(pid2docid, triples, triple_ids, train_time=True)

            train_binary_classification(args, retriever_model, optimizer, training_loader, verified_dev_loader=None)



        #TODO:checkpointing
        logger.info('checkpointing  model at {}'.format(args.model_file))
        ## check pointing ##
        save(args, retriever_model, optimizer, args.model_file + ".ckpt", epoch=stats['epoch'])
        #TODO:eval
        """logger.info("Evaluating on the full dev set....")
        top1 = eval_binary_classification(args, retriever_model, all_dev_exs, dev_loader, verified_dev_loader=None)
        if stats['best_acc'] < top1:
            stats['best_acc'] = top1
            logger.info('Best accuracy {}'.format(stats['best_acc']))
            logger.info('Saving model at {}'.format(args.model_file))
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
    parser.add_argument('-model_name', type=str, default='knn_index', help='Model name to load from as checkpoint')
    parser.add_argument('-cuda', type=bool, default=torch.cuda.is_available(), help='use cuda and gpu')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size to use')
    parser.add_argument('-data_workers', type=int, default=5, help='number of data workers to use')
    parser.add_argument('-training_folder', type=str, default='train/', help='folder with chunks of training triples')
    parser.add_argument('-num_training_files', type=int, default=10, help='number of chunks of training triples')
    parser.add_argument('-model_file', type=str, default='knn_index', help='Model file to store checkpoint')
    parser.add_argument('-out_dir', type=str, default='', help='Model file to store checkpoint')
    parser.add_argument('-pretrained', type=str, default='', help='checkpoint file to load checkpoint')


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
    args.model_file = os.path.join(args.out_dir, args.model_file)

    args.state_dict = None
    args.train = True

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    main(args)
