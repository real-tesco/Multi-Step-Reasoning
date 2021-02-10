import argparse
import logging

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', str2bool)

    # ranker model options
    parser.add_argument('-extra_layer', type=int, default=2500, help='dim of extra layer if 0 no layer will be added')
    parser.add_argument('-ranker_input', type=int, default=768,
                        help="dimension of the input to the ranker, should be twice of the embedding dim")
    parser.add_argument('-ranker_hidden', type=int, default=768,
                        help='hidden dimension of ranker')

    args = parser.parse_args()

    return args
