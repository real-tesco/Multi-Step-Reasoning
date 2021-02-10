import argparse
import json
import numpy as np


# preprocess the data into jsonl format for training and evaluation of models
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_trec', type=str)
    parser.add_argument('-input_qrels', type=str, default=None)
    parser.add_argument('-input_queries', type=str)
    parser.add_argument('-input_qids', type=str)
    parser.add_argument('-input_docs', type=str)
    parser.add_argument('-input_docids', type=str)
    parser.add_argument('-documents', type=int, default=0)
    parser.add_argument('-queries', type=int, default=0)
    parser.add_argument('-output', type=str)
    args = parser.parse_args()

    # for indexing documents/queries
    if args.documents > 0 or args.queries > 0:
        with open(args.input_docs) as f_in, open(args.output, "w") as f_out:
            cnt = 0
            for line in f_in:
                line = line.strip().split('\t')
                if args.documents > 0:
                    if len(line) != 4:
                        cnt += 1
                        print(f"Skipped: {line}, insg: {cnt}")
                    else:
                        id = line[0]
                        content = line[3]
                else:
                    if len(line) != 2:
                        cnt += 1
                        print(f"Skipped: {line}, insg: {cnt}")
                    else:
                        id = line[0]
                        content = line[1]

                f_out.write(json.dumps({'doc_id': id, 'doc': content}) + '\n')
    # for training/dev data
    else:
        qs = {}
        # generate dev data for ranker with embeddings
        if args.input_queries.split('.')[-1] == 'npy':
            queries = np.load(args.input_queries)
            qids = np.load(args.input_qids)
            for idx, _ in enumerate(queries):
                qs[str(qids[idx])] = queries[idx]

        elif args.input_queries.split('.')[-1] == 'json' or args.input_queries.split('.')[-1] == 'jsonl':
            with open(args.input_queries, 'r') as r:
                for line in r:
                    line = json.loads(line)
                    qs[line['query_id']] = line['query']
        else:
            with open(args.input_queries, 'r') as r:
                for line in r:
                    line = line.strip().split('\t')
                    qs[line[0]] = line[1]

        ds = {}
        # generate dev data for ranker with embeddings
        if args.input_docs.split('.')[-1] == 'npy':
            docs = np.load(args.input_docs)
            docids = np.load(args.input_docids)
            for idx, _ in enumerate(docids):
                ds[str(docids[idx])] = docs[idx]

        elif args.input_queries.split('.')[-1] == 'json' or args.input_queries.split('.')[-1] == 'jsonl':
            with open(args.input_docs, 'r') as r:
                for line in r:
                    line = json.loads(line)
                    ds[line['paper_id']] = ' '.join([line['title'], line['abstract']]).replace('\n', ' ').replace('\t', ' ').strip()
        else:
            with open(args.input_docs, 'r') as r:
                for line in r:
                    line = line.strip().split('\t')
                    if len(line) > 2:
                        ds[line[0]] = line[-2] + ' ' + line[-1]
                    else:
                        ds[line[0]] = line[1]

        if args.input_qrels is not None:
            qpls = {}
            with open(args.input_qrels, 'r') as r:
                for line in r:
                    line = line.strip().split()
                    if line[0] not in qpls:
                        qpls[line[0]] = {}
                    qpls[line[0]][line[2]] = int(line[3])

        f = open(args.output, 'w')
        with open(args.input_trec, 'r') as r:
            for line in r:
                line = line.strip().split()
                if line[0] not in qs or line[2] not in ds:
                    continue
                if args.input_qrels is not None:
                    if line[0] in qpls and line[2] in qpls[line[0]]:
                        label = qpls[line[0]][line[2]]
                        print("if true with label: ", label)
                    else:
                        label = 0
                    f.write(json.dumps({'query': qs[line[0]], 'doc': ds[line[2]], 'label': label, 'query_id': line[0], 'doc_id': line[2], 'retrieval_score': float(line[4])}) + '\n')
                else:
                    f.write(json.dumps({'query': qs[line[0]], 'doc': ds[line[2]], 'query_id': line[0], 'doc_id': line[2], 'retrieval_score': float(line[4])}) + '\n')
        f.close()


if __name__ == "__main__":
    main()
