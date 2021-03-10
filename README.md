# Query Modelling for Neural Retrieval

## Introduction
This git repository was created while pursuing my Master Thesis at the L3S institute at the Leibniz University Hannover.
I did ground work of exploring the Query Modelling for full neural retrieval without classic enhancements like Neural Retrieval combined with BM25 as ensemble.

The repository contains frameworks to train a Two Tower Model and generate an approximate KNN index on the MSMARCO document ranking dataset, as well as training a simple Ranker for the returned embeddings.

The key components in this work are the reformulators which can be trained and evaluated, aswell as the clustering approaches explored.

Feel free to clone this repository and explore this space even more.

## Setup
The requirements are in the requirements file and the environment file for conda. 
```
pip install -r requirements.txt
conda env create -f environment.yml
```

## Data
The pre-processed data, index files and checkpoints are available for download at our institute server so it is easier to get started. To get access just get in touch with us. Otherwise all used data can be generated using the preprocessing scripts. 
After un-taring, you will find a data directory containing all the data with the following structure:
```
embeddings/ -- containing the calculated embeddings in chunks (*.npy files)
indexes/ -- saved hnswlib knn index aswell as anserini index
checkpoints/ -- containg checkpoints of pretrained models, move to ../checkpoints/ dir
msmarco document ranking train/dev/test files
```

## Train Two Tower Document Encoder

You need to generate the training for the Document Encoder. It accepts a .tsv file with lines in following format: 
```
qid docid label
```

To generate the pairs training data you have to download the msmarco training, dev and test files. Then you can use following script:

```
python generate_train.py \
        -random_sample True \
        -pairs True \
        -negative_samples <number of negative samples per positive, here 4 negative samples were used>
        -base_dir <your base directory where the data is stored> \
        -out_file train-msmarco-pairs.tsv \
        -qrels msmarco-doctrain-qrels.tsv \
        -doc_lookup msmarco-docs-lookup.tsv \
```

You also have to preprocess the dev data into following jsonl format:

```
{"query": "<query text>", "doc": "<document text>", "label": <label>, "query_id": "<qid>, "doc_id": "<doc_id>", "retrieval_score", <retrieval_score>}
```

To do so use this script:

```
python preprocess.py \
        -output <msmarco-doc.dev.jsonl name of output file>
        -input_trec msmarco-docdev-top100 \
        -input qrels msmarco-docdev-qrels.tsv \
        -input_queries msmarco-docdev-queries.tsv \
        -input_docs msmarco-docs.tsv \
```

If you want to train new document embeddings use following script, it also outputs a tensorboard visualization for the training:
```
python train_retriever.py \
        -train  queries=./data/msmarco-doctrain-queries.tsv,docs=./data/msmarco-docs.tsv,qrels=./data/msmarco-doctrain-qrels.tsv,trec=./data/msmarco-train-pairs.tsv \        
        -save ./checkpoints/twotowerbert.bin \
        -dev ./data/msmarco-doc.dev.jsonl \
        -qrels ./data/msmarco-docdev-qrels.tsv \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -res ./results/two_tower_train.trec \
        -metric mrr_cut_100 \
        -max_query_len 64 \
        -max_doc_len 512 \
        -epoch 20 \
        -batch_size 32 \
        -lr 3e-6 \
        -n_warmup_steps 10000 \
        -eval_every 10000 \
        -print_every 250 \
        -tensorboard_output ./runs/train_two_tower
```

To store the learned embeddings as chunks preprocess the documents/queries as jsonl with format:
```
{"doc_id": <document/query id>, "doc": <document/query text>}
```

and run:

```
python train_retriever.py \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -max_query_len 64 \
        -max_doc_len 512 \
        -batch_size 32 \
        -two_tower_checkpoint ./checkpoints/twotowerbert.bin \
        -save_embeddings 1 \
        -embed ./data/msmarco-docs.jsonl \
        -docs_per_chunk 250000 \
        -embed_dir ./data/embeddings/ \
```
if you want to encode queries also use the flag and set max_doc_len to max_query_len: 
``` 
-embed_queries 1
-max_doc_len 64
-max_query_len 64
```

## Training Ranker

In this work no ranker was used. But a simple reranking architecture was implemented. To train the embedding ranker the documents and the queries need to be encoded as chunks by the retriever model. 
You can start training with following script: 

```
python train_ranker.py \
        -dev_query_embedding_file ./data/embeddings/marco_dev_query_embeddings_0.npy \
        -dev_query_ids_file ./data/embeddings/marco_dev_query_embeddings_indices_0.npy \
        -doc_embedding_format ./data/embeddings/marco_doc_embeddings_{}.npy \
        -doc_ids_format ./data/embeddings/marco_doc_embeddings_indices_{}.npy \
        -triples ./data/trids_marco-doc-10.tsv \
        -print_every 250 \
        -eval_every 10000
        -save ./checkpoints/ranker_extra_layer_2500.ckpt \
        -epochs 20 \
        -metric mrr_cut_100 \
        -extra_layer 2500
```
The ```-document_embedding_format``` and ```-doc_ids_format``` are the formats the document embeddings and their document ids are stored in chunks respectively.
The ```-dev_query_embedding_file``` and ```-dev_query_ids_file``` contain the encoded dev queries and their ids, also in .npy format.
The triples file contains the triples for the training of the ranker with format:

```qid \t pos_id \t neg_id```

Here I use 10 negative documents per query sampled from the top-100 BM25 documents, which are not in qrels file. The qrels file provided by MSMARCO only contains positive judged documents for a query.
To generate the file also use the generate_train script with following arguments:

```
python generate_train.py \
        -use_top_bm25_samples True \
        -doc_train_100_file msmarco-doctrain-top100 \
        -random_sample False \
        -negative_samples 10 \
        -pairs False \
        -base_dir <your base directory where the data is stored> \
        -out_file trids_marco-doc-10.tsv \
        -qrels msmarco-doctrain-qrels.tsv \
        -doc_lookup msmarco-docs-lookup.tsv \
```

## Training Reformulator

The pretrained reformulator models are in the downloaded checkpoints folder. If you want to train your own reformulator use following script: 

```
python train_reformulator.py \
        -dev_data ./data/msmarco-dev-queries-inference.jsonl \
        -train_data ./data/msmarco-train-queries-inference.jsonl \
        -print_every 100 \
        -model_name ./checkpoints/reformulator.bin \
        -epochs 20 \
        -metric mrr_cut_100 \
        -eval_every 10000 \
        -batch_size 32 \
        -k 1000 \
        -res ./results/reformulator.trec \
        -tensorboard_output ./boards/train_reformulator \
        -top_k_reformulator 10 \
        -reformulation_type [neural|weighted_avg|transformer] \
```
and chose one of the ```reformulator_type``` choices and set the corresponding hyper parameters respectively. Every reformulator considers the top top_k_reformulator documents of the retrieved set.

If reformulation_type is left out, default value is None, the base architecture with knn index and ranker is evaluated.

For the neural reformulator set hidden layer dimensions, if hidden2 is 0 don't use second hidden layer:
``` 
        -hidden1 3500 \
        -hidden2 0 \
```

For the transformer reformulator set the number of attention heads, the number of encoder layers and the dimension of the feedforward layer in each encoder layer:
``` 
        -nhead 4 \
        -num_encoder_layers 1 \
        -dim_feedforward 3072
``` 

For the weighted average there is no additionaly argument necessary. 

## Other Experiments 

With the inference.py script different experiments with the trained components are possible. The most important flags are:

```
-baseline True -- evaluates the BM25 baseline with chosen anserini index 
-exact_knn True -- evaluates the Two Tower model with exact knn, big matrix multiplication of all documents and queries
-ideal -- evaluates the ideal run where the reformulated query is an actualy relevant document, set number_ideal_samples > 1 to do the retrieval steps for a query with given number of relevant documents for the query
-print_embeddings True -- prints the embeddings and meta data of the first 3 queries to use on projector.tensorflow.org 
-test_clustering True -- clustering method to improve recall is tested, need to set -sampling to [cluster_kmeans|cluster_spectral|attention] 
```

*Acknowledgement*: This codebase started from [Multi-Step-Reasoning](https://github.com/rajarshd/Multi-Step-Reasoning). And a big thanks to Prof. Avishek Anand my first Reviewer and Jaspreet Singh my Tutor for all the help and suggestions you provided for me.


## Citation
```
@inproceedings{
2021querymodeling_neural_retrieval,
title={Query Modelling for Neural Retrieval},
author={L. J. Brandt},
booktitle={LUH},
year={2021},
}
```

