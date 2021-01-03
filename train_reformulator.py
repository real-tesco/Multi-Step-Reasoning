import argparse
import os
import torch
import torch.optim as optim
from torch.nn.functional import softmax
import msr
from msr.data.dataloader import DataLoader
from msr.data.datasets.rankingdataset import RankingDataset
from msr.data.datasets import BertDataset
from transformers import AutoTokenizer
from msr.knn_retriever.retriever import KnnIndex
from msr.knn_retriever.retriever_config import get_args as get_knn_args
from msr.knn_retriever.two_tower_bert import TwoTowerBert
from msr.reranker.ranking_model import NeuralRanker
from msr.reranker.ranker_config import get_args as get_ranker_args
import msr.utils as utils
from msr.reformulation.query_reformulation import NeuralReformulator, QueryReformulator, TransformerReformulator
import logging
import random
from transformers import get_linear_schedule_with_warmup


logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def inner_product(prediction, target):
    dot_prod = (prediction * target).sum(dim=1)
    return (1 - torch.sigmoid(dot_prod)).mean()


# other choice would be inner product
def cross_entropy(prediction, target):
    prediction = softmax(prediction, dim=1)
    target = softmax(target, dim=1)
    m = prediction.shape[0]
    log_likelihood = - (torch.log(prediction) * target)
    loss = log_likelihood.sum() / m
    return loss


def get_relevant_embeddings(qids, qrels, knn_index):
    targets = []
    for qid in qids:
        if qid in qrels:
            did = random.choice(qrels[qid])
            targets.append(knn_index.get_document(did))
        else:
            print(f"qid: {qid} is not in qrels...")
    #print(qrels.items())
    return torch.FloatTensor(targets)


def eval_pipeline(args, knn_index, ranking_model, reformulator, dev_loader, device):
    logger.info("Evaluating trec metrics for dev set...")
    if args.reformulation_type == 'weighted_avg':
        reformulator.layer.eval()
    else:
        reformulator.eval()
    rst_dict = {}
    for step, dev_batch in enumerate(dev_loader):

        query_id = dev_batch['query_id']
        with torch.no_grad():

            document_labels, document_embeddings, distances, query_embeddings = knn_index.knn_query_inference(
                dev_batch['q_input_ids'].to(device), dev_batch['q_segment_ids'].to(device), dev_batch['q_input_mask'].to(device))
            query_embeddings = query_embeddings.to(device)

            batch_score = ranking_model.rerank_documents(query_embeddings, document_embeddings.to(device),
                                                         device)
            #batch_score = batch_score.detach().cpu().tolist()

            # sort doc embeddings according score and reformulate
            sorted_scores, scores_sorted_indices = torch.sort(batch_score, dim=1, descending=True)
            sorted_docs = document_embeddings[
                torch.arange(document_embeddings.shape[0]).unsqueeze(-1), scores_sorted_indices].to(device)

            if args.reformulation_type == 'neural':
                new_queries = reformulator(query_embeddings, sorted_docs)
            elif args.reformulation_type == 'weighted_avg':
                new_queries = reformulator(sorted_docs, sorted_scores.to(device))
            elif args.reformulation_type == 'transformer':
                new_queries = reformulator(query_embeddings, sorted_docs)
            else:
                return

            # do another run with the reformulated queries
            document_labels, document_embeddings, distances, _ = knn_index.knn_query_embedded(
                new_queries.cpu())

            batch_score = ranking_model.rerank_documents(new_queries.to(device), document_embeddings.to(device),
                                                         device)
            batch_score = batch_score.detach().cpu().tolist()

            for (q_id, d_id, b_s) in zip(query_id, document_labels, batch_score):
                if q_id not in rst_dict:
                    rst_dict[q_id] = []
                for d, s in (zip(d_id, b_s)):
                    rst_dict[q_id].append((s, d))
        if (step + 1) % args.print_every == 0:
            logger.info(f"-- eval: {step + 1}/{len(dev_loader)} --")

    return rst_dict


def train(args, knn_index, ranking_model, reformulator, loss_fn, optimizer, m_scheduler, train_loader, dev_loader,
          qrels, metric, device, k=100):
    if args.reformulation_type == 'weighted_avg':
        reformulator.layer.train()
    else:
        reformulator.train()
    mrr = 0.0
    best_mrr = 0.0
    best_epoch = 0
    para_loss = utils.AverageMeter()
    for epoch in range(0, args.epochs):
        for idx, train_batch in enumerate(train_loader):
            if train_batch is None:
                print("None batch...")
                continue
            query_id = train_batch['query_id']
            document_labels, document_embeddings, distances, query_embeddings = knn_index.knn_query_inference(
                train_batch['q_input_ids'].to(device), train_batch['q_segment_ids'].to(device), train_batch['q_input_mask'].to(device))

            query_embeddings = query_embeddings.to(device)

            batch_score = ranking_model.rerank_documents(query_embeddings, document_embeddings.to(device), device)
            #batch_score = batch_score.detach().cpu().tolist()

            # sort doc embeddings according score and reformulate
            scores_sorted, scores_sorted_indices = torch.sort(batch_score, dim=1, descending=True)
            #scores_sorted, scores_sorted_indices = torch.sort(torch.tensor(batch_score), dim=1, descending=True)
            sorted_docs = document_embeddings[
                torch.arange(document_embeddings.shape[0]).unsqueeze(-1), scores_sorted_indices].to(device)

            # load relevant documents for current queries
            # new_queries should match document representation of relevant document
            target_embeddings = get_relevant_embeddings(query_id, qrels, knn_index).to(device)

            if args.reformulation_type == 'neural':
                new_queries = reformulator(query_embeddings, sorted_docs)
            elif args.reformulation_type == 'weighted_avg':
                new_queries = reformulator(sorted_docs, scores_sorted.to(device))
            elif args.reformulation_type == 'transformer':
                new_queries = reformulator(query_embeddings, sorted_docs)
            else:
                return

            # batch_loss = cross_entropy(new_queries, target_embeddings)
            # batch_loss = inner_product(new_queries, target_embeddings)
            batch_loss = loss_fn(new_queries, target_embeddings)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            m_scheduler.step()
            para_loss.update(batch_loss.data.item())

            if (idx + 1) % args.print_every == 0:
                logger.info('Epoch={} | iter={}/{} | avg loss={:2.4f} | last mrr={:2.5f} | '
                            'best mrr={:2.5f} ({})'.format(epoch, idx + 1, len(train_loader), para_loss.avg, mrr,
                                                           best_mrr, best_epoch))
                para_loss.reset()

            if (idx + 1) % args.eval_every == 0:
                with torch.no_grad():
                    rst_dict = eval_pipeline(args, knn_index, ranking_model, reformulator,
                                             dev_loader, device)
                    msr.utils.save_trec(args.res, rst_dict)
                    if args.metric.split('_')[0] == 'mrr':
                        mrr = metric.get_mrr(args.qrels, args.res, args.metric)
                    else:
                        mrr = metric.get_metric(args.qrels, args.res, args.metric)
                    if mrr > best_mrr:
                        msr.utils.save_trec(args.res + '.best', rst_dict)
                        best_mrr = mrr
                        best_epoch = epoch
                        logger.info('New best mes = {:2.4f}'.format(best_mrr))
                        if args.reformulation_type == 'weighted_avg':
                            torch.save(reformulator.layer.state_dict(), args.model_name)
                        else:
                            torch.save(reformulator.state_dict(), args.model_name)
        # Eval run and print
        _ = metric.eval_run(args.qrels, args.res + '.best')


def main():
    # setting args
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument('-reformulation_type',
                        type=str, default='neural', choices=['neural', 'weighted_avg', 'lstm', 'transformer'],
                        help='type of reformulator to train')
    parser.add_argument('-top_k_reformulator', type=int, default=5)
    parser.add_argument('-reformulator_checkpoint', type=str)

    # transformer reformulator args
    parser.add_argument('-nhead', type=int, default=4)
    parser.add_argument('-num_encoder_layers', type=int, default=1)
    parser.add_argument('-dim_feedforward', type=int, default=3072)

    # neural reformulator args
    parser.add_argument('-hidden1', type=int, default=1000)

    # training args
    parser.add_argument('-loss_fn', type=str, default='ip', help='loss function to use')

    # inference args
    parser.add_argument('-two_tower_checkpoint', type=str, default='./checkpoints/twotowerbert.bin')
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-ranker_checkpoint', type=str, default='./checkpoints/ranker_extra_layer_2500.ckpt')
    parser.add_argument('-dev_data', action=msr.utils.DictOrStr, default='./data/msmarco-dev-queries-inference.jsonl')
    parser.add_argument('-train_data', action=msr.utils.DictOrStr, default='./data/msmarco-train-queries-inference.jsonl')
    parser.add_argument('-qrels', type=str, default='./data/msmarco-docdev-qrels.tsv')
    parser.add_argument('-train_qrels', type=str, default='./data/msmarco-doctrain-qrels.tsv')
    parser.add_argument('-res', type=str, default='./results/reformulator.trec')
    parser.add_argument('-metric', type=str, default='mrr_cut_100')
    parser.add_argument('-batch_size', type=int, default='32')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-print_every', type=int, default=25)
    parser.add_argument('-eval_every', type=int, default=100)
    parser.add_argument('-train', type='bool', default=False)
    parser.add_argument('-full_ranking', type='bool', default=True)
    parser.add_argument('-reformulation_mode', type=str, default=None, choices=[None, 'top1', 'top5'])
    parser.add_argument('-k', type=int, default=100)
    parser.add_argument('-use_ranker_in_next_round', type='bool', default=True)
    parser.add_argument('-optimizer', type=str, default='adamax',
                        help='optimizer to use for training [sgd, adamax]')
    parser.add_argument('-lr', type=float, default=3e-6)
    parser.add_argument('-n_warmup_steps', type=int, default=10000)
    parser.add_argument('-weight_decay', type=float, default=0, help='Weight decay (default 0)')
    parser.add_argument('-momentum', type=float, default=0, help='Momentum (default 0)')
    parser.add_argument('-model_name', type=str, default='./checkpoints/reformulator.bin')

    # Legacy?
    parser.add_argument('-dataset', type=str, default='./data/reformulator_training_data.tsv')
    parser.add_argument('-query_embedding_format', type=str, default='./data/embeddings/embeddings_random_examples/'
                                                                     'marco_train_query_embeddings_{}.npy')
    parser.add_argument('-query_ids_format', type=str,
                        default='./data/embeddings/embeddings_random_examples/'
                                'marco_train_query_embeddings_indices_{}.npy')
    parser.add_argument('-dev_query_embedding_file', type=str, default='./data/marco_dev_query_embeddings_0.npy')
    parser.add_argument('-dev_query_ids_file', type=str, default='./data/marco_dev_query_embeddings_indices_0.npy')
    parser.add_argument('-dev_file', type=str, default='./data/msmarco-docdev-queries.tsv')
    parser.add_argument('-num_query_files', type=int, default=1)

    args = parser.parse_args()
    index_args = get_knn_args(parser)
    ranker_args = get_ranker_args(parser)
    ranker_args.train = False
    args.dim_hidden = index_args.dim_hidden

    tokenizer = AutoTokenizer.from_pretrained(index_args.pretrain)
    logger.info("Load training data...")
    train_dataset = BertDataset(
        dataset=args.train_data,
        tokenizer=tokenizer,
        mode='inference',
        query_max_len=index_args.max_query_len,
        doc_max_len=index_args.max_doc_len,
        max_input=args.max_input
    )
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8)

    # DataLoaders for dev
    logger.info("Loading dev data...")
    dev_dataset = BertDataset(
        dataset=args.dev_data,
        tokenizer=tokenizer,
        mode='inference',
        query_max_len=index_args.max_query_len,
        doc_max_len=index_args.max_doc_len,
        max_input=args.max_input
    )
    dev_loader = DataLoader(dev_dataset, args.batch_size, shuffle=False, num_workers=8)

    '''
    # Refactor to use real queries
    logger.info("Loading train data...")
    doc_embedding_list = []
    doc_ids_list = []
    query_embedding_list = [args.query_embedding_format.format(i) for i in range(0, args.num_query_files)]
    query_ids_list = [args.query_ids_format.format(i) for i in range(0, args.num_query_files)]

    train_dataset = RankingDataset(doc_embedding_list, doc_ids_list, query_embedding_list, query_ids_list,
                                   dataset=args.dataset, mode='train', model='reformulator')
    train_loader = msr.data.dataloader.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )
    
    logger.info("Loading dev data...")
    dev_query_embedding_list = [args.dev_query_embedding_file]
    dev_query_ids_list = [args.dev_query_ids_file]

    dev_dataset = RankingDataset(doc_embedding_list, doc_ids_list, dev_query_embedding_list, dev_query_ids_list,
                                   dataset=args.dev_file, mode='dev', model='reformulator')
    dev_loader = msr.data.dataloader.DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8
    )
    '''
    # Load qrels for target document embedding
    qrels = {}
    with open(args.train_qrels, "r") as f:
        for line in f:
            qid, _, did, label = line.strip().split()
            if int(label) > 0:
                if qid in qrels:
                    qrels[qid].append(did)
                else:
                    qrels[qid] = [did]

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # Loading models
    #    1. Load Retriever
    logger.info("Loading Retriever...")
    two_tower_bert = TwoTowerBert(index_args.pretrain)
    checkpoint = torch.load(args.two_tower_checkpoint)
    two_tower_bert.load_state_dict(checkpoint)
    knn_index = KnnIndex(index_args, two_tower_bert)
    logger.info("Load Index File and set ef")
    knn_index.load_index()
    knn_index.set_ef(index_args.efc)
    # Transformer to big if all is on GPU
    knn_index.set_device(device)

    #   2. Load Ranker
    logger.info("Loading Ranker...")
    ranking_model = NeuralRanker(ranker_args)
    checkpoint = torch.load(args.ranker_checkpoint)
    ranking_model.load_state_dict(checkpoint)
    ranking_model.to(device)

    #   3. Load Reformulator
    logger.info('Loading Reformulator...')
    if args.reformulation_type == 'neural':
        reformulator = NeuralReformulator(args.top_k_reformulator, args.dim_hidden, args.hidden1)
        reformulator.to(device)
    elif args.reformulation_type == 'weighted_avg':
        reformulator = QueryReformulator(mode='weighted_avg', topk=args.top_k_reformulator)
        reformulator.layer.to(device)
    elif args.reformulation_type == 'transformer':
        reformulator = TransformerReformulator(args.top_k_reformulator, args.nhead, args.num_encoder_layers,
                                               args.dim_feedforward)
        reformulator.to(device)
        if torch.cuda.device_count() > 1:
            logger.info(f'Using DataParallel with {torch.cuda.device_count()} GPUs...')
            reformulator = torch.nn.DataParallel(reformulator)
    else:
        return

    if args.reformulator_checkpoint:
        logger.info("loading checkpoint...")
        checkpoint = torch.load(args.reformulator_checkpoint)

        if args.reformulation_type == 'weighted_avg':
            reformulator.layer.load_state_dict(checkpoint)
        else:
            reformulator.load_state_dict(checkpoint)

    # set loss_fn
    if args.loss_fn == 'ip':
        loss_fn = inner_product
    elif args.loss_fn == 'cross_entropy':
        loss_fn = cross_entropy

    # set optimizer and scheduler
    if args.reformulation_type == 'weighted_avg':
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, reformulator.layer.parameters()), lr=args.lr)
    else:
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, reformulator.parameters()), lr=args.lr)
    m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps,
                                                  num_training_steps=len(train_dataset) * args.epochs // args.batch_size)

    # set metric
    metric = msr.metrics.Metric()

    # starting inference
    logger.info("Starting training...")
    train(args, knn_index, ranking_model, reformulator, loss_fn, m_optim, m_scheduler, train_loader, dev_loader, qrels, metric, device, args.k)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    file_handler = logging.FileHandler('./train_rformulator.log', mode='w', encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    main()
