import os
import torch
import msr
from msr.data.datasets.rankingdataset import RankingDataset
from msr.reranker.ranking_model import NeuralRanker
import msr.reranker.utilities as utils
import msr.reranker.ranker_config as config
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


def make_dataloader(doc_list, docid_list, query_list, query_id_list, triples, mode='train'):
    dataset = RankingDataset(doc_list, docid_list, query_list, query_id_list, triples, mode=mode)
    loader = msr.data.dataloader.DataLoader(
        dataset,
        batch_size=args.batch_size if mode == 'train' else args.batch_size * 8,
        shuffle=(mode == 'train'),
        num_workers=args.data_workers
    )
    return loader


def save(args, model, optimizer, filename):
    params = {'state_dict': {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }}
    try:
        torch.save(params, filename)
    except BaseException:
        logger.warn('[ WARN: Saving failed... continuing anyway. ]')


def init_from_checkpoint(args):
    logger.info('Loading model from saved checkpoint {}'.format(args.pretrained))
    checkpoint = torch.load(args.pretrained)
    ranker = NeuralRanker(args)
    ranker.load_state_dict(checkpoint['state_dict']['model'])

    parameters = ranker.parameters()
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

    return ranker, optimizer


def init_from_scratch(args):
    ranker = NeuralRanker(args)
    parameters = ranker.parameters()

    optimizer = None
    if parameters is not None:
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

    return ranker, optimizer


# TODO: look here
def train(args, loss, ranking_model, optimizer, device, train_loader, dev_loader):
    para_loss = utils.AverageMeter()
    best_mrr = 0.0
    mrr = 0.0
    for epoch in range(0, args.epochs):
        for idx, ex in enumerate(train_loader):
            if ex is None:
                continue
            logger.info(f'{idx} in epoch {epoch}')
            logger.info('before calculation')
            scores_p, scores_n = ranking_model.score_documents(ex['query'].to(device),
                                                               ex['positive_doc'].to(device),
                                                               ex['negative_doc'].to(device))  # todo: look here
            logger.info('after calculation')
            batch_loss = loss(scores_p, scores_n, torch.ones(scores_p.size()).to(device))
            logger.info('after loss')
            optimizer.zero_grad()
            batch_loss.backward()
            #torch.nn.utils.clip_grad_norm(ranking_model.parameters(), 2.0)
            optimizer.step()
            para_loss.update(batch_loss.data.item())
            logger.info('after optimizer step')
            if math.isnan(para_loss.avg):
                import pdb
                pdb.set_trace()

            if idx+1 % args.print_every == 0:
                logger.info('Epoch = {} | iter={}/{} | avg loss = {:2.4f} | last mrr = {} |current best mrr = {}'.format(
                    stats['epoch'],
                    idx, len(train_loader),
                    para_loss.avg,
                    mrr,
                    best_mrr))
                para_loss.reset()

            if idx+1 % args.eval_every == 0:
                mrr = eval_ranker(args, ranking_model, dev_loader, device)
                if mrr > best_mrr:
                    best_mrr = mrr
                    logger.info('New best MRR = {:2.4f}'.format(mrr))
                    logger.info('checkpointing  model at {}.ckpt'.format(args.model_name))
                    save(args, ranking_model, optimizer, args.model_name + ".ckpt")


def eval_ranker(args, model, dev_loader, device):
    logger.info("Evaluating trec metrics for dev set...")
    rst_dict = {}
    model.train = False
    for step, dev_batch in enumerate(dev_loader):
        # TODO: msmarco dataset refactoring
        query_id, doc_id, label, retrieval_score = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label'], \
                                                   dev_batch['retrieval_score']
        with torch.no_grad():

            batch_score, _, _ = model.score_documents(dev_batch['query'].to(device),
                                                      dev_batch['doc'].to(device))

            # TODO: write to file
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id in rst_dict:
                    rst_dict[q_id].append((b_s, d_id, l))
                else:
                    rst_dict[q_id] = [(b_s, d_id, l)]
        if (step + 1) % args.print_every == 0:
            print(f"-- eval: {step + 1}/{len(dev_loader)} --")
    model.train = True
    utils.save_trec(args.out_file, rst_dict)
    logger.info("Done with evaluation, use trec_eval to evaluate run...")
    mrr = utils.get_mrr(args.qrels, args.out_file)
    return mrr


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Loading train data...")
    doc_embedding_list = (args.doc_embedding_format.format(i) for i in range(0, args.num_doc_files))
    doc_ids_list = (args.doc_ids_format.format(i) for i in range(0, args.num_doc_files))
    query_embedding_list = (args.query_embedding_format.format(i) for i in range(0, args.num_query_files))
    query_ids_list = (args.query_ids_format.format(i) for i in range(0, args.num_query_files))

    train_loader = make_dataloader(doc_embedding_list, doc_ids_list, query_embedding_list, query_ids_list, args.triples,
                                   mode='train')

    logger.info("Loading dev data...")
    doc_embedding_list = (args.doc_embedding_format.format(i) for i in range(0, args.num_doc_files))
    doc_ids_list = (args.doc_ids_format.format(i) for i in range(0, args.num_doc_files))
    dev_query_embedding_list = [args.dev_query_embedding_file]
    dev_query_ids_list = [args.dev_query_ids_file]
    dev_loader = make_dataloader(doc_embedding_list, doc_ids_list, dev_query_embedding_list,
                                 dev_query_ids_list, args.dev_file, mode='dev')

    # initialize Model
    if args.checkpoint:
        logger.info('Initializing model from checkpoint...')
        ranker_model, optimizer = init_from_checkpoint(args)
    else:
        logger.info('Initializing model from scratch...')
        ranker_model, optimizer = init_from_scratch(args)

    ranker_model.to(device)
    loss = torch.nn.MarginRankingLoss(margin=1)
    loss = loss.to(device)

    if args.train:
        logger.info("Starting training...")
        train(args, loss, ranker_model, optimizer, device, train_loader, dev_loader)


if __name__ == '__main__':
    args = config.get_args()

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    main(args)
