import torch
import time
import logging
import subprocess
import shlex
import sys
import os

logger = logging.getLogger()

def batchify(args, train_time, index_time):
    return lambda x: batchify_(args, x, train_time, index_time)


def batchify_(args, batch, train_time, index_time):
    """Gather a batch of individual examples into one batch."""

    new_batch = []
    for d in batch:
        if d is not None:
            new_batch.append(d)
    batch = new_batch
    if len(batch) == 0:
        return None
    if train_time:
        #qid, pid, nid, query, positive, negative

        qids = [ex[0] for ex in batch]
        pids = [ex[1] for ex in batch]
        nids = [ex[2] for ex in batch]
        queries = [ex[3] for ex in batch]
        positives = [ex[4] for ex in batch]
        negatives = [ex[5] for ex in batch]

        q = torch.FloatTensor(len(queries), len(queries[0]))
        for idx, query in enumerate(queries):
            q[idx].copy_(query)

        p = torch.FloatTensor(len(positives), len(positives[0]))
        for idx, positive in enumerate(positives):
            p[idx].copy_(positive)

        n = torch.FloatTensor(len(negatives), len(negatives[0]))
        for idx, negative in enumerate(queries):
            n[idx].copy_(negative)

        return q, p, n, qids, pids, nids
    if index_time:
        pids = [ex[0] for ex in batch]
        passages = [ex[1] for ex in batch]
        p = torch.FloatTensor(len(passages), len(passages[0]))
        for idx, passage in enumerate(passages):
            p[idx].copy_(passage)
        return p, pids

class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_run_with_trec_eval(qrels, prediction, path_to_trec_eval):
    cmd = os.path.join(path_to_trec_eval, f" trec_eval {qrels} {prediction} -m map -m recip_rank -m ndcg")
    pargs = shlex.split(cmd)
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pout, perr = p.communicate()
    print("running {}".format(cmd))
    if perr != '':
        print("error occured! : {}".format(perr))
    if sys.version_info[0] < 3:
        lines = pout.split('\n')
    else:
        lines = pout.split(b'\n')
    MAP = float(lines[0].strip().split()[-1])
    MRR = float(lines[1].strip().split()[-1])
    ndcg = float(lines[2].strip().split()[-1])

    logger.info(f"MAP: {MAP}")
    logger.info(f"MRR: {MRR}")
    logger.info(f"MRR: {ndcg}")
    print(f"MAP: {MAP}")
    print(f"MRR: {MRR}")
    print(f"MRR: {ndcg}")
    return MAP, MRR, ndcg