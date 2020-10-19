import torch
import time
import logging

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

    qids = [ex[0] for ex in batch]
    pids = [ex[1] for ex in batch]
    nids = [ex[2] for ex in batch]
    queries = [ex[3] for ex in batch]
    positives = [ex[4] for ex in batch]
    negatives = [ex[5] for ex in batch]

    #logger.info(f"DEBUG: {qids}")

    queries = torch.as_tensor(queries)
    positives = torch.as_tensor(positives)
    negatives = torch.as_tensor(negatives)

    return queries, positives, negatives, qids, pids, nids


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
