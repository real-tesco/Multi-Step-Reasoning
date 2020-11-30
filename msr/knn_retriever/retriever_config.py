import argparse
import os
import logging
import torch
import msr

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', str2bool)

    parser.add_argument('-pretrain', type=str, default='bert-base-uncased')
    parser.add_argument('-max_query_len', type=int, default=20)
    parser.add_argument('-max_doc_len', type=int, default=150)
    parser.add_argument('-M', type=int, default=64)
    parser.add_argument('-efc', type=int, default=100)
    parser.add_argument('-similarity', type=str, default='ip')
    parser.add_argument('-dim_hidden', type=int, default=768)
    args = parser.parse_args()

    return args
