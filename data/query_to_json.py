import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_queries', type=str)
    parser.add_argument('-output', type=str)
    args = parser.parse_args()

    qs = {}
    f = open(args.output, 'w')
    with open(args.input_queries, 'r') as r:
        for line in r:
            line = line.strip().split('\t')
            qs[line[0]] = line[1]
            f.write(json.dumps(
                {'query_id': line[0], 'query': line[2]}) + '\n')
        f.close()


if __name__ == "__main__":
    main()
