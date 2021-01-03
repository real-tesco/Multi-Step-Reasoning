from typing import List, Dict
import numpy as np
import pytrec_eval
import matplotlib.pyplot as plt


class Metric():
    def get_metric(self, qrels: str, trec: str, metric: str = 'ndcg_cut_10') -> Dict[str, float]:
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
            mes[measure] = pytrec_eval.compute_aggregated_measure(measure, [query_measures[measure] for query_measures in results.values()])
        return mes[metric]

    def get_mrr(self, qrels: str, trec: str, metric: str = 'mrr_cut_100') -> float:
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
                    rr = 1 / (i+1)
                    break
            mrr += rr
        mrr /= len(run)
        return mrr

    def get_mrr_dict(self, qrels: str, trec: str, metric: str = 'mrr_cut_100'):
        k = int(metric.split('_')[-1])
        query_mrr_dict = {}
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
            query_mrr_dict[qid] = rr
            mrr += rr
        mrr /= len(run)
        return mrr, query_mrr_dict

    def eval_run(self, qrels: str, trec: str, print_graphs=False) -> Dict[str, float]:
        query_measure_to_check = ['ndcg', 'map', 'P_10', 'recall_5', 'recall_10', 'recall_30', 'recall_100']
        with open(qrels, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)
        with open(trec, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
        results = evaluator.evaluate(run)
        for query_id, query_measures in sorted(results.items()):
            pass
        mes = {}
        for measure in sorted(query_measure_to_check):
            mes[measure] = pytrec_eval.compute_aggregated_measure(measure, [query_measures[measure] for query_measures in results.values()])
        mrr, mrr_dict = self.get_mrr_dict(qrels, trec)

        # std of mrr values
        mrr_values = [mrr_dict[qid] for qid in mrr_dict]
        std_dev = np.std(mrr_values)

        # calc number of unjudged documents returned
        cnt = 0
        total = 0
        for qid in run:
            for docid in run[qid]:
                if docid not in qrel[qid]:
                    cnt += 1
                total += 1

        mes['mrr'] = mrr
        for key in mes:
            print("{}: {:.4f}".format(key, mes[key]))
        print(f"\nstd_dev of mrr: {std_dev:2.4f}\n")
        print(f"Unjudged documents in result: {cnt}/{total}")
        print(f"Unjudged documents avg per query: {cnt / len(run)}/{total / len(run)}")

        if print_graphs:
            query_measure_to_print = ['recall_100', 'recall_30', 'ndcg', 'map', 'P_10', 'mrr_cut_100']

            # 6 ticks on axis
            buckets = [[[] for _ in range(6)] for _ in range(len(query_measure_to_print))]

            for query_id, query_measures in sorted(results.items()):
                # print(type(query_measures))
                # for measure, value in sorted(query_measures.items()):
                for i, measure in enumerate(query_measure_to_print):
                    if measure == 'mrr_cut_100':
                        value = mrr_dict[query_id]
                    else:
                        value = query_measures[measure]
                    if value <= 0.1:
                        buckets[i][0].append(query_id)
                    elif value <= 0.3:
                        buckets[i][1].append(query_id)
                    elif value <= 0.5:
                        buckets[i][2].append(query_id)
                    elif value <= 0.7:
                        buckets[i][3].append(query_id)
                    elif value <= 0.9:
                        buckets[i][4].append(query_id)
                    else:
                        buckets[i][5].append(query_id)
            xs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            ys = [[len(b[i]) / len(results) for i in range(len(b))] for b in buckets]

            fig = plt.figure(figsize=(10, 10))

            sub1 = plt.subplot(2, 3, 1)
            sub2 = plt.subplot(2, 3, 2)
            sub3 = plt.subplot(2, 3, 3)
            sub4 = plt.subplot(2, 3, 4)
            sub5 = plt.subplot(2, 3, 5)
            sub6 = plt.subplot(2, 3, 6)

            plots = [sub1, sub2, sub3, sub4, sub5, sub6]

            for i, sub in enumerate(plots):
                sub.plot(xs, ys[i], 'b.')
                sub.set_title(query_measure_to_print[0])
                sub.set_xticks(xs)
                sub.set_xlabel('measure value bucket')
                sub.set_ylabel('fraction in bucket')

            fig.suptitle(trec, fontsize=14)
            plt.savefig(trec + '.png')

        return mes

