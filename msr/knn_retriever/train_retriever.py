import os

import argparse
import torch
from msmarco import MSMARCO
from retriever import KnnIndex
import utilities as utils
import retriever_config as config
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch.optim as optim
import numpy as np
import logging
import json
from torch.autograd import Variable
import math
import hnswlib


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


logger = logging.getLogger()

global_timer = utils.Timer()
stats = {'timer': global_timer, 'epoch': 0, 'recall': 0.0}


def make_dataloader(queries, qids, pid2docid, triples, triple_ids, passages, pids, train_time=False, dev_time=False,
                    index_time=False):
    dataset = MSMARCO(queries, qids, pid2docid, triples, triple_ids, passages, pids, train_time=train_time,
                      dev_time=dev_time, index_time=index_time)
    sampler = SequentialSampler(dataset) if not train_time and not index_time else RandomSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.data_workers,
        collate_fn=utils.batchify(args, train_time=train_time, index_time=index_time),  # TODO: write batch function for dataloader
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
    # args = checkpoint['args']
    args.state_dict = checkpoint['state_dict']
    ret = KnnIndex(args)
    logger.info(f"ARGS.TRAIN: {args.train}")
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


# TODO: look here
def train_binary_classification(args, ret_model, optimizer, train_loader):
    para_loss = utils.AverageMeter()
    ret_model.query_transformer.train()
    ret_model.document_transformer.train()
    for idx, ex in enumerate(train_loader):
        if ex is None:
            continue

        inputs = [e if e is None or type(e) != type(ex[0]) else Variable(e.cuda())
                  for e in ex[:3]]
        ret_input = [*inputs[:]]

        scores_positive, scores_negative = ret_model.score_documents(*ret_input)  # todo: look here

        loss = torch.nn.MarginRankingLoss(margin=0.4, reduction='sum')
        target = torch.ones_like(scores_positive)

        batch_loss = loss(scores_positive, scores_negative, target)
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
            logger.info('Epoch = {} | iter={}/{} | avg loss = {:2.4f}\n'
                        '__________________________________________________________ \n'
                        'Positive Scores = {} \n'
                        'Negative Scores = {} \n'
                        '__________________________________________________________'.format(
                stats['epoch'],
                idx + stats['chunk'] * len(train_loader), len(train_loader) * args.num_training_files,
                para_loss.avg,
                torch.sum(scores_positive),
                torch.sum(scores_negative)))
            para_loss.reset()


def test_index(args):

    logger.info("Load index")
    index = hnswlib.Index(space=args.similarity, dim=args.dim_hidden)
    logger.info(args.hnsw_index)
    index.load_index(args.hnsw_index)

    logger.info('Evaluate self-recall on first chunk...')
    model = args.model
    current_passage_file = os.path.join(args.passage_folder, "msmarco_passages_normedf32_0.npy")
    current_index_file = os.path.join(args.passage_folder, "msmarco_indices_0.npy")
    with open(current_passage_file, "rb") as f:
        chunk = torch.from_numpy(np.load(f)).cuda()
    with open(current_index_file, "rb") as f:
        indices = np.load(f)
    chunk = model.query_transformer.forward(chunk)
    labels, distances = index.knn_query(chunk.cpu().detach().numpy(), k=1)
    logger.info("Recall for dataset: ", np.mean(labels.reshape(labels.shape[0]) == indices))
    logger.info("Evaluating recall for dev set...")
    with open(args.dev_queries, "rb") as f:
        dev_queries = torch.from_numpy(np.load(f)).cuda()
    with open(args.dev_qids, "rb") as f:
        dev_qids = np.load(f)
    dev_queries = model.query_transformer.forward(dev_queries)
    labels, distances = index.knn_query(dev_queries.cpu().detach().numpy(), k=100)
    with open(args.outfile, "w") as f:
        for idx, qid in enumerate(dev_qids):
            ranked_docids = []
            for idy, (label, distance) in enumerate(zip(labels[idx], distances[idx])):
                docid = args.pid2docid_dict[label]
                if docid in ranked_docids:
                    continue
                f.write("{} Q0 {} {} {} {}".format(qid, docid, idy, distance, f"M{args.M}EFC{args.efc}"))
                ranked_docids.append(docid)
    logger.info("Done with evaluation, use trec_eval to evaluate run...")
    _, _, _ = utils.evaluate_run_with_trec_eval(args.qrels_file, args.outfile, args.trec_eval)


def build_index(args):
    model = args.model
    index_name = os.path.join(args.out_dir, "msmarco_knn_M_{}_efc_{}.bin".format(args.M, args.efc))
    if os.path.isfile(index_name):
        index = hnswlib.load_index(index_name)
        start = args.start_chunk
    else:
        logger.info('Initializing index with parameters:\n'
                    'Max_elements={}\n'
                    'ef_construction={}\n'
                    'M={}\n'
                    'dimension={}\n'.format(args.max_elems, args.efc, args.M, args.dim_hidden))
        index = hnswlib.Index(space=args.similarity, dim=args.dim_hidden)
        index.init_index(max_elements=args.max_elems, ef_construction=args.efc, M=args.M)
        start = 0

    for i in range(start, args.num_passage_files):
        current_passage_file = os.path.join(args.passage_folder, "msmarco_passages_normedf32_" + str(i) + ".npy")
        current_index_file = os.path.join(args.passage_folder, "msmarco_indices_" + str(i) + ".npy")
        with open(current_passage_file, "rb") as f:
            chunk = np.load(f)
        with open(current_index_file, "rb") as f:
            indices = np.load(f)
        index_loader = make_dataloader(None, None, None, None, None, chunk, indices, index_time=True)

        for idx, ex in enumerate(index_loader):
            if ex is None:
                continue

            inputs = [e if e is None or type(e) != type(ex[0]) else Variable(e.cuda())
                      for e in ex[:1]]
            passages = model.document_transformer.forward(inputs[0])

            index.add_items(passages.cpu().detach().numpy(), ex[1])
        logger.info("Added {}/{} chunks...".format(i+1, args.num_passage_files))
    index.save_index(index_name)

    logger.info("Finished saving index with name: {}".format(index_name))
    #args.hnsw_index = index
    #return args


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
        retriever_model, optimizer = init_from_checkpoint(args)
    else:
        logger.info('Initializing model from scratch...')
        retriever_model, optimizer = init_from_scratch(args)
    if args.train:
        logger.info("Starting training...")
        for epoch in range(0, args.epochs):
            stats['epoch'] = epoch
            # need to load the training data in chunks since its too big
            for i in range(0, args.num_training_files):
                logger.info("Load current chunk of training data...")
                triples = np.load(os.path.join(args.training_folder, "train.triples_msmarco" + str(i) + ".npy"))
                triple_ids = np.load(os.path.join(args.training_folder, "msmarco_indices_" + str(i) + ".npy"))
                stats['chunk'] = i
                training_loader = make_dataloader(None, None, pid2docid, triples, triple_ids, None, None,
                                                  train_time=True)
                train_binary_classification(args, retriever_model, optimizer, training_loader)

            logger.info('checkpointing  model at {}.ckpt'.format(args.model_file))
            save(args, retriever_model, optimizer, args.model_file + ".ckpt", epoch=stats['epoch'])
        save(args, retriever_model, optimizer, args.model_file + ".max", epoch=stats['epoch'])

    retriever_model.document_transformer.eval()
    retriever_model.query_transformer.eval()
    args.model = retriever_model
    args.train = False

    if args.index:
        build_index(args)
    if args.test:
        test_index(args)


if __name__ == '__main__':
    args = config.get_args()

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    main(args)
