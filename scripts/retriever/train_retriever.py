import argparse
import torch
from msr.knn_retriever.msmarco import MSMARCO
from msr.knn_retriever import
from torch.utils.data.sampler import SequentialSampler, RandomSampler


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def make_dataloader(passages, pids, queries, qids, pid2docid, train_time=False):
    dataset = MSMARCO(passages, pids, queries, qids, pid2docid)
    sampler = SequentialSampler(dataset) if not train_time else RandomSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify(args, args.para_mode, train_time=train_time), #TODO: write batch function for dataloader
        pin_memory=True
    )

def main(args):
    # initialize Model
    # load training data with data loaders
    for i in range(0, args.epochs):
        #train for 1 epoch
        #evaluation for this epoch
        #propagate loss back though network


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument('-similarity', type=str, default='ip', choices=['cosine', 'l2', 'ip'],
                        help='similarity score to use when knn index is chosen')
    parser.add_argument('-dimension', type=int, default=768,
                        help='dimension of the embeddings for knn index')
    parser.add_argument('-path_to_index', type=str,
                        help='path to the hnswlib vector index')
    parser.add_argument('-optimizer', type=str, default='sgd',
                        help='optimizer to use for training [sgd, adamax]')
    parser.add_argument('-epochs', type=int, default=30,
                        help='number of epochs to train the retriever')
    args = parser.parse_args()
    main(args)
