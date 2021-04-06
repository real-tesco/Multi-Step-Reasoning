import os
import argparse
import logging
import json
from itertools import product

import hnswlib
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import msr as msr
import msr.utils as utils
from msr.knn_retriever.retriever_config import get_args


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


logger = logging.getLogger()

global_timer = utils.Timer()
stats = {'timer': global_timer, 'epoch': 0, 'recall': 0.0}


def dev(args, model, dev_loader, device):
    logger.info("starting evaluation...")
    rst_dict = {}
    for step, dev_batch in enumerate(dev_loader):
        query_id, doc_id, label, retrieval_score = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label'], \
                                                   dev_batch['retrieval_score']
        with torch.no_grad():
            batch_score, _, _ = model(dev_batch['q_input_ids'].to(device),
                                      dev_batch['d_input_ids'].to(device),
                                      dev_batch['q_input_mask'].to(device),
                                      dev_batch['q_segment_ids'].to(device),
                                      dev_batch['d_input_mask'].to(device),
                                      dev_batch['d_segment_ids'].to(device))

            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id in rst_dict:
                    rst_dict[q_id].append((b_s, d_id, l))
                else:
                    rst_dict[q_id] = [(b_s, d_id, l)]
        if (step+1) % (4*args.print_every) == 0:
            logger.info(f"-- Evaluation: {step+1}/{len(dev_loader)} --")
    return rst_dict


def train(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, device, writer):
    best_mes = 0.0
    mes = 0.0
    for epoch in range(args.epoch):
        avg_loss = 0.0
        for step, train_batch in enumerate(train_loader):
            inputs = (train_batch['q_input_ids'].to(device),
                      train_batch['d_input_ids'].to(device),
                      train_batch['q_input_mask'].to(device),
                      train_batch['q_segment_ids'].to(device),
                      train_batch['d_input_mask'].to(device),
                      train_batch['d_segment_ids'].to(device))

            batch_score, queries, documents = model(*inputs)
            print("batchscore: ", batch_score.shape)
            print("qs: ", queries.shape)
            print("ds: ", documents.shape)
            if args.loss_fn == 'bce':
                batch_loss = loss_fn(batch_score.float(), train_batch['label'].float().to(device))
                if torch.cuda.device_count() > 1:
                    batch_loss = batch_loss.mean()
            elif args.loss_fn == 'ranknet':
                batch_score_negatives = torch.mm(queries, documents.t())
                batch_loss = loss_fn(batch_score, batch_score_negatives, device)
            elif args.loss_fn == 'margin_ranking':
                positives, negatives = prepare_margin(batch_score, queries, documents, device)
                batch_loss = loss_fn(positives, negatives, torch.ones_like(positives))

            avg_loss += batch_loss.item()
            batch_loss.backward()
            m_optim.step()
            m_scheduler.step()
            m_optim.zero_grad()

            if (step + 1) % args.print_every == 0:
                logger.info(f"Epoch={epoch} | {step + 1} / {len(train_loader)} | {avg_loss / args.print_every} | "
                            f"last metric: {mes} | best metric: {best_mes}")
                writer.add_scalar('training loss', avg_loss / args.print_every, epoch * len(train_loader) + step)
                avg_loss = 0.0

            if (step + 1) % args.eval_every == 0:
                with torch.no_grad():
                    rst_dict = dev(args, model, dev_loader, device)
                    msr.utils.save_trec(args.res, rst_dict)
                    if args.metric.split('_')[0] == 'mrr':
                        mes, _ = metric.get_mrr_dict(args.qrels, args.res, args.metric)
                    else:
                        mes = metric.get_metric(args.qrels, args.res, args.metric)
                if mes >= best_mes:
                    best_mes = mes
                    msr.utils.save_trec(args.res + ".best", rst_dict)
                    logger.info('save_model...')
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), args.save)
                    else:
                        torch.save(model.state_dict(), args.save)


def save_embeddings(args, model, doc_loader, device):
    model.eval()
    with torch.no_grad():
        docs = []
        doc_ids = []
        chunk = 0
        embed_type = "query" if args.embed_queries else "doc"
        embed_docs = not args.embed_queries
        for idx, document_dict in enumerate(doc_loader):
            doc_id = document_dict['doc_id']
            doc_embedding = model.calculate_embedding(document_dict['d_input_ids'].to(device),
                                                      document_dict['d_input_mask'].to(device),
                                                      document_dict['d_segment_ids'].to(device), doc=embed_docs)
            docs.extend(doc_embedding.detach().cpu().tolist())
            doc_ids.extend(doc_id)
            if len(docs) == args.docs_per_chunk or (idx+1) == len(doc_loader):
                docs = np.asarray(docs).astype(np.float32)
                logger.info(f'shape of docs: {docs.shape}')
                np.save(os.path.join(args.embed_dir, 'marco_' + embed_type + '_embeddings_' + str(chunk) +
                                     '.npy'), docs)
                doc_ids = np.asarray(doc_ids)
                np.save(os.path.join(args.embed_dir, 'marco_' + embed_type + '_embeddings_indices_' + str(chunk) +
                                     '.npy'), doc_ids)
                chunk += 1
                docs = []
                doc_ids = []
                logger.info('New chunk saved to disk...')
            if (idx+1) % args.print_every == 0:
                logger.info(f' {idx+1} / {len(doc_loader)} | calculated {(idx + 1) * args.batch_size} embeddings '
                            f'| saved in {chunk} chunks..')


def test_index(args):
    logger.info("Load index")
    index = hnswlib.Index(space=args.similarity, dim=args.dim_hidden)
    logger.info(args.index_file)
    index.load_index(args.index_file)

    logger.info('Evaluate self-recall on first chunk...')
    current_passage_file = os.path.join(args.embed_dir, "marco_doc_embeddings_0.npy")
    with open(current_passage_file, "rb") as f:
        chunk = np.load(f)
        lbls = np.arange(len(chunk))
    labels, distances = index.knn_query(chunk, k=1)
    logger.info("Recall for dataset encoded with doc transformer: "
                "{}".format(np.mean(labels.reshape(labels.shape[0]) == lbls)))


def build_index(args):
    if args.reverse_passage:
        name = "firstP_and_remainP"
    else:
        name = "firstP"
    index_name = os.path.join(args.out_dir, "msmarco_{}_512_knn_M_{}_efc_{}.bin".format(name, args.M, args.efc))
    if os.path.isfile(index_name):
        logger.info('Loading index with parameters: \n'
                    'ef_construction={}\n'
                    'M={}\n'
                    'dimension={}'.format(args.M, args.efc, args.dim_hidden)
                    )
        index = hnswlib.Index(space=args.similarity, dim=args.dim_hidden)
        index.load_index(index_name, max_elements=args.max_elems)
        with open(args.index_mapping, "r") as f:
            docid2index = json.load(f)
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
        docid2index = {}

    for i in range(start, args.num_passage_files):

        current_passage_file = os.path.join(args.embed_dir, "marco_doc_embeddings_" + str(i) + ".npy")
        current_index_file = os.path.join(args.embed_dir, "marco_doc_embeddings_indices_" + str(i) + ".npy")
        index_dataset = msr.data.IndexDataset(current_passage_file, current_index_file)
        index_loader = msr.data.DataLoader(dataset=index_dataset, shuffle=False,
                                           batch_size=args.batch_size, num_workers=8)

        for idx, ex in enumerate(index_loader):
            if ex is None:
                continue

            doc_ids, docs = ex['id'], ex['doc']
            # indices in hnswlib index are numbered from 0 to max_elements
            indices = np.arange(i * args.num_per_passage + (idx * args.batch_size),
                                i * args.num_per_passage + (idx + args.batch_size) + len(doc_ids))

            index.add_items(docs, indices)
            for idy, did in enumerate(doc_ids):
                if did in docid2index:
                    docid2index[did] = [docid2index[did], indices[idy]]
                else:
                    docid2index[did] = indices[idy]
        logger.info("Added {}/{} chunks...".format(i + 1, args.num_passage_files))
    index.save_index(index_name)
    with open(args.index_mapping, "w") as f:
        json.dump(docid2index, f)
    logger.info("Finished saving index with name: {}".format(index_name))


def test_bert_checkpoint(args, model, metric, dev_loader, device):
    rst_dict = dev(args, model, dev_loader, device)
    msr.utils.save_trec(args.res, rst_dict)
    _ = metric.eval_run(args.qrels, args.res)


def prepare_margin(pos_scores, queries, documents, device):
    # pos_scores [batchsize x 1]
    #neg_scores = torch.empty(queries.shape[0], queries.shape[0]-1).to(device)
    positives = []
    negatives = []
    for i in range(queries.shape[0]):
        for j in range(documents.shape[0]):
            if i == j:
                continue
            else:
                negatives.append((queries[i] * documents[j]).sum())
                positives.append(pos_scores[i])

    positives = torch.tensor(positives)
    negatives = torch.tensor(negatives)
    print("neg_shape: ", negatives.shape)
    print("pos_shape: ", positives.shape)
    print("pos: ", positives)
    print("negs: ", negatives)
    return positives, negatives


def ranknet(y_pred_pos, y_pred_neg, device):
    diff = torch.empty(y_pred_pos.shape[0], y_pred_neg.shape[0]).to(device)
    for i in range(y_pred_pos.shape[0]):
        for j in range(y_pred_neg.shape[0]):
            diff[i, j] = y_pred_pos[i] - y_pred_neg[i, j]
    loss = torch.log(1 + torch.exp(diff)).mean(-1).mean(-1)
    return loss


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.vocab)
    if args.save_embeddings > 0:
        logger.info('reading documents to embed..')
        if args.reverse_passage:
            passage_type = 'reverse'
        else:
            passage_type = 'normal'
        embed_set = msr.data.datasets.BertDataset(
            dataset=args.embed,
            tokenizer=tokenizer,
            mode='embed',
            passage_type=passage_type,
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,  # 3213835,
        )
        embed_loader = msr.data.DataLoader(
            dataset=embed_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8
        )
    elif not args.index:
        logger.info('reading training data...')
        train_set = msr.data.datasets.BertDataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
        )
        logger.info('reading dev data...')
        dev_set = msr.data.datasets.BertDataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input
        )

        logger.info("creating loaders...")
        train_loader = msr.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8
        )
        dev_loader = msr.data.DataLoader(
            dataset=dev_set,
            batch_size=args.batch_size * 16,
            shuffle=False,
            num_workers=8
        )
    logger.info("loading model...")

    model = msr.knn_retriever.TwoTowerBert(pretrained=args.pretrain, projection_dim=args.projection_dim)

    if args.two_tower_checkpoint is not None:
        logger.info("Loading model from checkpoint")
        state_dict = torch.load(args.two_tower_checkpoint)
        model.load_state_dict(state_dict, strict=False)
    elif args.bert_checkpoint is not None:
        logger.info("Loading model from checkpoint")
        state_dict = torch.load(args.bert_checkpoint)
        model.load_state_dict(state_dict)
        # model.load_bert_model_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.learn_projection:
        logger.info("Freezing two tower BERTs, only learn projection layer")
        model.freeze_berts()

    metric = msr.metrics.Metric()

    if args.test_bert:
        test_bert_checkpoint(args, model, metric, dev_loader, device)

    if args.save_embeddings > 0:
        save_embeddings(args, model, embed_loader, device)

    elif args.index:
        build_index(args)

    elif args.test_index:
        test_index(args)

    else:
        logger.info(f"Loss_fn: {args.loss_fn}")
        if args.loss_fn == 'bce':
            loss_fn = torch.nn.BCELoss()
            loss_fn.to(device)
        elif args.loss_fn == 'ranknet':
            loss_fn = ranknet
        elif args.loss_fn == 'margin_ranking':
            loss_fn = torch.nn.MarginRankingLoss()

        if torch.cuda.device_count() > 1:
            logger.info(f'Using DataParallel with {torch.cuda.device_count()} GPUs...')
            loss_fn = torch.nn.DataParallel(loss_fn)
            model = torch.nn.DataParallel(model)

        if args.freeze_doc_tower:
            logger.info("Freezing document BERT")
            model.freeze_document_tower()

        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps,
                                                      num_training_steps=len(train_set) * args.epoch // args.batch_size)

        writer = SummaryWriter(args.tensorboard_output)

        logger.info("starting training...")
        train(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, device, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument('-train', action=msr.utils.DictOrStr, default='./data/train_toy.jsonl')
    parser.add_argument('-tensorboard_output', type=str)
    parser.add_argument('-max_input', type=int, default=12800000)
    parser.add_argument('-save', type=str, default='./checkpoints/twotowerbert.bin')
    parser.add_argument('-dev', action=msr.utils.DictOrStr, default='./data/dev_toy.jsonl')
    parser.add_argument('-qrels', type=str, default='./data/qrels_toy')
    parser.add_argument('-vocab', type=str, default='bert-base-uncased')
    parser.add_argument('-two_tower_checkpoint', type=str, default=None)
    parser.add_argument('-bert_checkpoint', type=str, default=None)
    parser.add_argument('-res', type=str, default='./results/twotowerbert.trec')
    parser.add_argument('-metric', type=str, default='mrr_cut_100')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-loss_fn', type=str, default='bce', choices=['bce', 'ranknet', 'margin_ranking'])
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-n_warmup_steps', type=int, default=10000)
    parser.add_argument('-eval_every', type=int, default=10000)
    parser.add_argument('-print_every', type=int, default=1000)
    parser.add_argument('-embed', type=str, default='./data/msmarco-docs.jsonl')
    parser.add_argument('-save_embeddings', type='bool', default=False)
    parser.add_argument('-test_bert', type='bool', default=False)
    parser.add_argument('-embed_queries', type='bool', default=False)
    parser.add_argument('-embed_dir', type=str, default='./data/embeddings')
    parser.add_argument('-index', type='bool', default=False)
    parser.add_argument('-docs_per_chunk', type=int, default=200000)
    parser.add_argument('-out_dir', type=str, default='./results')
    parser.add_argument('-start_chunk', type=int, default=0)
    parser.add_argument('-reverse_passage', type='bool', default=False)
    parser.add_argument('-max_elems', type=int)
    parser.add_argument('-learn_projection', type='bool', default=False)
    parser.add_argument('-projection', type=int, default=0)
    parser.add_argument('-freeze_doc_tower', type='bool', default=False)
    parser.add_argument('-test_index', type='bool', default=False)

    args = get_args(parser)

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    file_handler = logging.FileHandler('./train_retriever.log', mode='w', encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    main(args)
