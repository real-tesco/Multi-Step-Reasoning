import msr.metrics.metric as metric
import argparse
import os
import sys
import pytrec_eval
import msr.metrics.metric


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('qrel')
    parser.add_argument('run')

    args = parser.parse_args()

    assert os.path.exists(args.qrel)
    assert os.path.exists(args.run)

    metrics = msr.metrics.metric.Metric()

    _ = metrics.eval_run(args.qrel, args.run)


if __name__ == "__main__":
    sys.exit(main())
