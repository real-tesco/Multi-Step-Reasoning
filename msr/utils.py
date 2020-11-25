import os
import json
from argparse import Action
import time
import Dict


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


def save_trec(rst_file, rst_dict):
    with open(rst_file, 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores, key=lambda x: x[0], reverse=True)
            for rank, value in enumerate(res):
                writer.write(q_id+' Q0 '+str(value[1])+' '+str(rank+1)+' '+str(value[0])+' twotower\n')
    return


def get_metric(qrels: str, trec: str, metric: str = 'ndcg_cut_10') -> Dict[str, float]:
    with open(qrels, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    with open(trec, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run)
    for query_id, query_measures in sorted(results.items()):
        pass
    mes = {}
    for measure in sorted(query_measures.keys()):
        mes[measure] = pytrec_eval.compute_aggregated_measure(measure, [query_measures[measure] for query_measures in
                                                                        results.values()])
    return mes[metric]


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