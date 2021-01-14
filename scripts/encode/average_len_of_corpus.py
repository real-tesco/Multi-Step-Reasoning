import argparse
from tqdm import tqdm
import nltk
import json

parser = argparse.ArgumentParser()
parser.add_argument('corpus')
args = parser.parse_args()

with open(args.corpus, "r") as input_f:
    avg_len = 0
    number_of_docs = 0
    for line in tqdm(input_f):
        if args.corpus.split(".")[-1] == "tsv":
            _, _, _, content = line.split('\t')
        elif args.corpus.split(".")[-1] == "jsonl":
            j = json.loads(line)
            content = j["contents"]
        else:
            print("use .tsv or .jsonl file...")
            exit(0)
        #  += len(content.split(" ")) # faster
        avg_len += len(nltk.word_tokenize(content))
        number_of_docs += 1
    print(f"Average length of given dataset: {avg_len/number_of_docs}")
