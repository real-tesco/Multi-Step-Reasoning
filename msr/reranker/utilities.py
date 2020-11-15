import torch
import time
import logging
import subprocess
import shlex
import sys
import os

logger = logging.getLogger()


def batchify(args, train_time):
    return lambda x: batchify_(args, x, train_time)


def batchify_(args, batch, train_time):
    """Gather a batch of individual examples into one batch."""

    new_batch = []
    for d in batch:
        if d is not None:
            new_batch.append(d)
    batch = new_batch
    if len(batch) == 0:
        return None

    #qid, pid, nid, query, positive, negative
    #qids = [ex[0] for ex in batch]
    #pids = [ex[1] for ex in batch]
    #nids = [ex[2] for ex in batch]
    queries = [ex[0] for ex in batch]
    positives = [ex[1] for ex in batch]
    negatives = [ex[2] for ex in batch]
    q = torch.FloatTensor(len(queries), len(queries[0]))
    for idx, query in enumerate(queries):
        q[idx].copy_(query)
    p = torch.FloatTensor(len(positives), len(positives[0]))
    for idx, positive in enumerate(positives):
        p[idx].copy_(positive)
    n = torch.FloatTensor(len(negatives), len(negatives[0]))
    for idx, negative in enumerate(queries):
        n[idx].copy_(negative)
    return q, p, n#, qids, pids, nids


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


def save_trec(rst_file, rst_dict):
    with open(rst_file, 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores, key=lambda x: x[0], reverse=True)
            for rank, value in enumerate(res):
                writer.write(q_id+' Q0 '+str(value[1])+' '+str(rank+1)+' '+str(value[0])+' openmatch\n')
    return


def get_mrr(qrels: str, trec: str, metric: str = 'mrr_cut_10') -> float:
    k = int(metric.split('_')[-1])
    qrel = {}
    with open(qrels, 'r') as f_qrel:
        for line in f_qrel:
            qid, _, did, label = line.strip().split()
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][did] = int(label)
    run = {}
    with open(trec, 'r') as f_run:
        for line in f_run:
            qid, _, did, _, _, _ = line.strip().split()
            if qid not in run:
                run[qid] = []
            run[qid].append(did)
    mrr = 0.0
    for qid in run:
        rr = 0.0
        for i, did in enumerate(run[qid][:k]):
            if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
                rr = 1 / (i + 1)
                break
        mrr += rr
    mrr /= len(run)
    return mrr


def evaluate_run_with_trec_eval(qrels, prediction, path_to_trec_eval):
    logger.info(f"path to trec eval file: {path_to_trec_eval}")
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