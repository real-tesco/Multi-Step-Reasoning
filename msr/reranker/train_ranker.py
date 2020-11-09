import os
import torch
from msmarco import MSMARCO
from ranking_model import NeuralRanker
import utilities as utils
import ranker_config as config
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


def make_dataloader(pid2docid, triples, triple_ids, train_time=False, dev_time=False):
    dataset = MSMARCO(pid2docid, triples, triple_ids, train_time=train_time, dev_time=dev_time)
    sampler = SequentialSampler(dataset) if not train_time else RandomSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.data_workers,
        collate_fn=utils.batchify(args, train_time=train_time),
        # TODO: write batch function for dataloader
        pin_memory=True
    )
    return loader


def save(args, model, optimizer, filename, epoch=None):
    params = {'state_dict': {
        'model': model.state_dict(),
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


# TODO: look here
def train_binary_classification(args, ranking_model, optimizer, train_loader):
    para_loss = utils.AverageMeter()
    ranking_model.train()

    for idx, ex in enumerate(train_loader):
        if ex is None:
            continue

        inputs = [e if e is None or type(e) != type(ex[0]) else Variable(e.cuda())
                  for e in ex[:3]]
        ranker_input = [*inputs[:]]

        scores_positive, scores_negative = ranking_model.score_documents(*ranker_input)  # todo: look here
        true_labels_positive = torch.ones_like(scores_positive)
        true_labels_negative = torch.zeros_like(scores_negative)

        loss = torch.nn.BCEWithLogitsLoss()
        batch_loss = loss(scores_positive, true_labels_positive)
        batch_loss += loss(scores_negative, true_labels_negative)

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm(ranking_model.parameters(), 2.0)
        optimizer.step()
        para_loss.update(batch_loss.data.item())

        if math.isnan(para_loss.avg):
            import pdb
            pdb.set_trace()

        if idx % 25 == 0 and idx > 0:
            logger.info('Epoch = {} | iter={}/{} | avg loss = {:2.4f}\n'.format(
                stats['epoch'],
                idx + stats['chunk'] * len(train_loader), len(train_loader) * args.num_training_files,
                para_loss.avg))
            para_loss.reset()


def eval_ranker(args):
    logger.info("Load Model")

    logger.info("Evaluating trec metrics for dev set...")

    with open(args.dev_queries, "rb") as f:
        dev_queries = torch.from_numpy(np.load(f)).cuda()
    with open(args.dev_qids, "rb") as f:
        dev_qids = np.load(f)

    with open(args.out_file, "w") as f:
        for idx, qid in enumerate(dev_qids):
            ranked_docids = []
            current_rank = 1
            for idy, (label, distance) in enumerate(zip(labels[idx], distances[idx])):
                docid = args.pid2docid_dict[str(label)]
                if docid in ranked_docids:
                    continue
                f.write("{} Q0 {} {} {} {}\n".format(qid, docid, current_rank, 1.0 - distance, args.hnsw_index))
                current_rank += 1
                ranked_docids.append(docid)
    logger.info("Done with evaluation, use trec_eval to evaluate run...")
    # _, _, _ = utils.evaluate_run_with_trec_eval(args.qrels_file, args.out_file, args.trec_eval)


def main(args):
    # load data from files
    logger.info('Starting load data...')
    logger.info(f'using cuda: {args.cuda}')
    logger.info(f'args train: {args.train}')

    with open(args.pid2docid, 'r') as f:
        pid2docid = json.load(f)
        args.pid2docid_dict = pid2docid

    # initialize Model
    if args.checkpoint:
        logger.info('Initializing model from checkpoint...')
        ranker_model, optimizer = init_from_checkpoint(args)
    else:
        logger.info('Initializing model from scratch...')
        ranker_model, optimizer = init_from_scratch(args)
    if args.cuda:
        ranker_model.cuda()
    if args.train:
        logger.info("Starting training...")
        for epoch in range(0, args.epochs):
            stats['epoch'] = epoch
            # need to load the training data in chunks since its too big
            for i in range(0, args.num_training_files):
                logger.info("Load current chunk of training data...")
                if args.num_training_files > 1:
                    triples = np.load(os.path.join(args.training_folder, "train.triples_msmarco" + str(i) + ".npy"))
                    #triple_ids = np.load(os.path.join(args.training_folder, "msmarco_indices_" + str(i) + ".npy"))
                    triple_ids = None
                    stats['chunk'] = i
                else:
                    triples = np.load(os.path.join(args.training_folder, "train.triples_msmarco.npy"))
                    triple_ids = None
                training_loader = make_dataloader(pid2docid, triples, triple_ids, train_time=True)
                train_binary_classification(args, ranker_model, optimizer, training_loader)

            logger.info('checkpointing  model at {}.ckpt'.format(args.model_name))
            save(args, ranker_model, optimizer, args.model_name + ".ckpt", epoch=stats['epoch'])
        save(args, ranker_model, optimizer, args.model_name + ".max", epoch=stats['epoch'])
    if args.eval:
        eval_ranker(args)


if __name__ == '__main__':
    args = config.get_args()

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logger.info(f"path after join: {args.trec_eval}")
    main(args)
