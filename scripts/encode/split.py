#!/usr/bin/python3
import numpy as np

with open("./input/msmarco_indices_0.npy", "rb") as f:
    indices = np.load(f)

with open("./input/train.triples_msmarco0.npy", "rb") as f:
    triples = np.load(f)

np.save("./input/indices_small.npy", indices[:100])
np.save("./input/triples_small.npy", triples[:100])
