import os
import time
from argparse import Action


class DictOrStr(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if '=' in values:
            my_dict = {}
            for kv in values.split(","):
                k, v = kv.split("=")
                my_dict[k] = v
            setattr(namespace, self.dest, my_dict)
        else:
            setattr(namespace, self.dest, values)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# only used in inference, takes care of duplicate documents in final ranked list
# rst_dict has qid as key and list of (doc_id, score)
def save_trec_inference(rst_file, rst_dict):
    with open(rst_file, 'w') as writer:
        cnt = 0
        for q_id, scores in rst_dict.items():
            res = sorted(scores, key=lambda x: x[1], reverse=True)
            ranked_docids = []
            current_rank = 1
            for rank, value in enumerate(res):
                if value[0] not in ranked_docids:
                    writer.write(q_id+' Q0 '+str(value[0])+' '+str(current_rank)+' '+str(value[1])+' twotower_full\n')
                    ranked_docids.append(value[0])
                    current_rank += 1
                else:
                    cnt += 1
        print("Number avg deleted duplicate doids: ", cnt / len(rst_dict))
    return


# used in training of models, doesnt care if duplicates are in result list
# CARE: rst_dict has keys of qid and value of (score, doc_id)
def save_trec(rst_file, rst_dict):
    with open(rst_file, 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores, key=lambda x: x[0], reverse=True)
            for rank, value in enumerate(res):
                writer.write(q_id+' Q0 '+str(value[1])+' '+str(rank+1)+' '+str(value[0])+' twotower_embed\n')
    return


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