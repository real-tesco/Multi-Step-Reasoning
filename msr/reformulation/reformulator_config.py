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

    parser.add_argument('-reformulation_mode', type=str, default=None, choices=[None, 'top1', 'top5', 'weighted_avg', 'transformer'])
    parser.add_argument('-top_k_reformulator', type=int, default=10)

    # transformer
    parser.add_argument('-nhead', type=int, default=6)
    parser.add_argument('-num_encoder_layers', type=int, default=84)
    parser.add_argument('-dim_feedforward', type=int, default=500)

    #neural
    parser.add_argument('-dim_embedding', type=str, default='ip')
    parser.add_argument('-hidden1', type=int, default=768)


    args = parser.parse_args()

    return args
