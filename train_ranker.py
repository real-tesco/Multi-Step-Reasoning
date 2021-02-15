import argparse
import torch
import torch.optim as optim
import logging
import math

import msr
import msr.utils as utils
import msr.reranker.ranker_config as config
from msr.data.datasets.rankingdataset import RankingDataset
from msr.reranker.ranking_model import NeuralRanker


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


logger = logging.getLogger()
global_timer = utils.Timer()
stats = {'timer': global_timer, 'epoch': 0, 'recall': 0.0}


def make_dataloader(doc_list, docid_list, query_list, query_id_list, triples, mode='train', model='ranker'):
    dataset = RankingDataset(doc_list, docid_list, query_list, query_id_list, triples, mode=mode, model=model)
    loader = msr.data.dataloader.DataLoader(
        dataset,
        batch_size=args.batch_size if mode == 'train' else args.batch_size * 8,
        shuffle=(mode == 'train'),
        num_workers=args.data_workers
    )
    return loader


def init_from_checkpoint(args):
    logger.info('Loading model from saved checkpoint {}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    ranker = NeuralRanker(args)
    ranker.load_state_dict(checkpoint)

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


def train(args, loss, ranking_model, metric, optimizer, device, train_loader, dev_loader):
    para_loss = utils.AverageMeter()
    mes = 0.0
    best_mes = 0.0
    best_ndcg = 0.0
    ndcg = 0.0
    for epoch in range(0, args.epochs):
        for idx, ex in enumerate(train_loader):
            if ex is None:
                continue
            scores_p, scores_n = ranking_model.score_documents(ex['query'].to(device),
                                                               ex['positive_doc'].to(device),
                                                               ex['negative_doc'].to(device))

            batch_loss = loss(scores_p, scores_n, torch.ones(scores_p.size()).to(device))

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            para_loss.update(batch_loss.data.item())

            if math.isnan(para_loss.avg):
                import pdb
                pdb.set_trace()

            if (idx + 1) % args.print_every == 0:
                logger.info('Epoch={} | iter={}/{} | avg loss={:2.4f} | last mes={:2.5f} | '
                            'best mes={:2.5f} | last ndcg={:2.5f} | best ndcg={:2.5f}'.format(
                    epoch,
                    idx + 1, len(train_loader),
                    para_loss.avg,
                    mes,
                    best_mes,
                    ndcg,
                    best_ndcg))
                para_loss.reset()

            if (idx + 1) % args.eval_every == 0:
                with torch.no_grad():
                    rst_dict = eval_ranker(args, ranking_model, dev_loader, device)
                    msr.utils.save_trec(args.res, rst_dict)
                    if args.metric.split('_')[0] == 'mrr':
                        mes = metric.get_mrr(args.qrels, args.res, args.metric)
                    else:
                        mes = metric.get_metric(args.qrels, args.res, args.metric)
                    if mes > best_mes:
                        msr.utils.save_trec(args.res + '.best', rst_dict)
                        best_mes = mes
                        logger.info('New best metric = {:2.4f}'.format(best_mes))
                        logger.info('checkpointing  model at {}.ckpt'.format(args.save))
                        torch.save(ranking_model.state_dict(), args.save + ".ckpt")

        # eval at the end of each epoch
        _ = metric.eval_run(args.qrels, args.res + '.best')


def process_batch(model, batch, rst_dict, device):
    query_id, doc_id = batch['query_id'], batch['doc_id']
    with torch.no_grad():
        batch_score = model.score_documents(batch['query'].to(device),
                                            batch['doc'].to(device))

        batch_score = batch_score.detach().cpu().tolist()
        for (q_id, d_id, b_s) in zip(query_id, doc_id, batch_score):
            if q_id not in rst_dict:
                rst_dict[q_id] = [(b_s[0], d_id)]
            else:
                rst_dict[q_id].append((b_s[0], d_id))


def eval_ranker(args, model, dev_loader, device, test_loader=None):
    logger.info("Evaluating trec metrics for dev set...")
    rst_dict_dev = {}
    rst_dict_test = None
    model.eval()

    if test_loader:
        logger.info("Evaluating trec metrics for test set")
        rst_dict_test = {}
        for step, test_batch in enumerate(test_loader):
            process_batch(model, test_batch, rst_dict_test, device)

            if (step + 1) % args.print_every == 0:
                print(f"-- eval: {step + 1}/{len(test_loader)} --")

    for step, dev_batch in enumerate(dev_loader):
        process_batch(model, dev_batch, rst_dict_dev, device)
        if (step + 1) % args.print_every == 0:
            print(f"-- eval: {step + 1}/{len(dev_loader)} --")

    model.train()
    return rst_dict_dev, rst_dict_test


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train:
        logger.info("Loading train data...")

        # ranker training/dev data is in form of triple with ids and doc/query embeddings and ids as chunked numpy files
        doc_embedding_list = (args.doc_embedding_format.format(i) for i in range(0, args.num_doc_files))
        doc_ids_list = (args.doc_ids_format.format(i) for i in range(0, args.num_doc_files))
        query_embedding_list = (args.query_embedding_format.format(i) for i in range(0, args.num_query_files))
        query_ids_list = (args.query_ids_format.format(i) for i in range(0, args.num_query_files))

        train_loader = make_dataloader(doc_embedding_list, doc_ids_list, query_embedding_list, query_ids_list,
                                       args.triples,
                                       mode='train', model='ranker')

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

    metric = msr.metrics.Metric()

    if args.train:
        loss = torch.nn.MarginRankingLoss(margin=1.0)
        loss = loss.to(device)
        logger.info("Starting training...")
        train(args, loss, ranker_model, metric, optimizer, device, train_loader, dev_loader)
    elif args.eval:
        logger.info("Loading dev data...")
        doc_embedding_list = (args.doc_embedding_format.format(i) for i in range(0, args.num_doc_files))
        doc_ids_list = (args.doc_ids_format.format(i) for i in range(0, args.num_doc_files))
        dev_query_embedding_list = [args.dev_query_embedding_file]
        dev_query_ids_list = [args.dev_query_ids_file]
        dev_loader = make_dataloader(doc_embedding_list, doc_ids_list, dev_query_embedding_list,
                                     dev_query_ids_list, args.dev_data, mode='test')

        logger.info("loading test data")
        doc_embedding_list = (args.doc_embedding_format.format(i) for i in range(0, args.num_doc_files))
        doc_ids_list = (args.doc_ids_format.format(i) for i in range(0, args.num_doc_files))
        test_query_embedding_list = [args.test_query_embedding_file]
        test_query_ids_list = [args.test_query_ids_file]
        test_loader = make_dataloader(doc_embedding_list, doc_ids_list, test_query_embedding_list,
                                      test_query_ids_list, args.test_data, mode='test')

        dev_dict, test_dict = eval_ranker(args, ranker_model, dev_loader, device, test_loader)

        utils.save_trec(args.res + '.dev', dev_dict)
        utils.save_trec(args.res + '.test', test_dict)
        logger.info("Results for dev: ")
        _ = metric.eval_run(args.qrels, args.res + '.dev')
        logger.info("Results for test: ")
        _ = metric.eval_run(args.qrels_test, args.res + '.test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.register('type', 'bool', str2bool)

    # run options
    parser.add_argument('-train', type='bool', default=True, help='train document ranker')
    parser.add_argument('-eval', type='bool', default=False, help='eval ranker ')
    parser.add_argument('-print_every', type=int, default=25)
    parser.add_argument('-eval_every', type=int, default=10000)
    parser.add_argument('-cuda', type=bool, default=torch.cuda.is_available(), help='use cuda and gpu')
    parser.add_argument('-data_workers', type=int, default=5, help='number of data workers to use')
    parser.add_argument('-res', type=str, default='./results/ranking_result.trec',
                        help='metric to evaluate ranker with')

    # training options
    parser.add_argument('-epochs', type=int, default=30,
                        help='number of epochs to train the retriever')
    parser.add_argument('-weight_decay', type=float, default=0, help='Weight decay (default 0)')
    parser.add_argument('-learning_rate', type=float, default=0.1, help='Learning rate for SGD (default 0.1)')
    parser.add_argument('-momentum', type=float, default=0, help='Momentum (default 0)')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size to use')
    parser.add_argument('-optimizer', type=str, default='adamax',
                        help='optimizer to use for training [sgd, adamax]')
    parser.add_argument('-checkpoint', type=str, default=None, help='Checkpoint name of ranker model')
    parser.add_argument('-save', type=str, default='./checkpoints/ranker', help='path to store ranker model')
    parser.add_argument('-metric', type=str, default='ndcg_cut_10', help='metric to evaluate ranker with')

    # data settings
    parser.add_argument('-qrels', type=str, default='./data/msmarco-docdev-qrels.tsv',
                        help='dev qrels file')
    parser.add_argument('-qrels_test', type=str, default='./data/msmarco-test-qrels.tsvs')
    parser.add_argument('-doc_embedding_format', type=str,
                        default='./data/embeddings/embeddings_random_examples/marco_doc_embeddings_{}.npy',
                        help='folder with chunks of document embeddings, with format brackets for idx')
    parser.add_argument('-doc_ids_format', type=str,
                        default='./data/embeddings/embeddings_random_examples/marco_doc_embeddings_indices_{}.npy',
                        help='folder with chunks of document ids, with format brackets for idx')
    parser.add_argument('-num_doc_files', type=int, default=13, help='number of chunks of training triples')
    parser.add_argument('-query_embedding_format', type=str,
                        default='./data/embeddings/embeddings_random_examples/marco_train_query_embeddings_{}.npy',
                        help='folder with chunks of document embeddings, with format brackets for idx')
    parser.add_argument('-query_ids_format', type=str,
                        default='./data/embeddings/embeddings_random_examples/marco_train_query_embeddings_indices_{}.npy',
                        help='folder with chunks of document ids, with format brackets for idx')
    parser.add_argument('-num_query_files', type=int, default=1, help='number of chunks of training triples')
    parser.add_argument('-triples', type=str, default='./data/trids_marco-doc-10.tsv',
                        help='number of chunks of training triples')
    parser.add_argument('-dev_file', type=str, default='./data/msmarco-doc.dev.jsonl')
    parser.add_argument('-dev_query_embedding_file', type=str,
                        default='./data/embeddings/embeddings_random_examples/marco_dev_query_embeddings_0.npy')
    parser.add_argument('-dev_query_ids_file', type=str,
                        default='./data/embeddings/embeddings_random_examples/marco_dev_query_embeddings_indices_0.npy')
    parser.add_argument('-dev_data', type=str, default='./data/results/inference_bm25_baseline_1000.trec.dev')
    parser.add_argument('-test_query_embedding_file', type=str,
                        default='./data/embeddings/embeddings_random_examples/marco_dev_query_embeddings_0.npy')
    parser.add_argument('-test_query_ids_file', type=str,
                        default='./data/embeddings/embeddings_random_examples/marco_dev_query_embeddings_indices_0.npy')
    parser.add_argument('-test_data', type=str, default='./results/inference_bm25_baseline_1000.trec.test')

    args = config.get_args(parser)

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    file_handler = logging.FileHandler('./train_retriever.log', mode='w', encoding='utf-8')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_handler)

    main(args)
