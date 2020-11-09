#!/usr/bin/python3
import numpy as np
from tqdm import tqdm

with open("/home/brandt/msmarco/trids_marco-doc-10.tsv", "r") as f, \
        open("/home/brandt/msmarco/train-marco-doc.tsv", "w") as out:
    already_done_positive_example_for_qid = -1
    for line in tqdm(f):
        qid, posid, negid = line.split()
        if qid == already_done_positive_example_for_qid:
            out.write(f"{qid}\t{negid}\t0\n")
        else:
            out.write(f"{qid}\t{posid}\t1\n")
            out.write(f"{qid}\t{negid}\t0\n")
            already_done_positive_example_for_qid = qid

