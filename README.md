# Multi Step Query Modelling for Document Retrieval


*Acknowledgement*: This codebase started from [Multi-Step-Reasoning](https://github.com/rajarshd/Multi-Step-Reasoning).

## Setup
The requirements are in the requirements file and the environment file for conda. 
```
pip install -r requirements.txt
conda env create -f environment.yml
```

## Data
The pre-processed data, index files and checkpoints are available for download so it is easier to get started. They can be downloaded from [here](http://iesl.cs.umass.edu/downloads/multi-step-reasoning-iclr19/data.tar.gz).
After un-taring, you will find a data directory containing all the data with the following structure:
```
embeddings/ -- containing the calculated embeddings in chunks (*.npy files)
indexes/ -- saved hnswlib knn index aswell as anserini index
checkpoints/ -- containg checkpoints of pretrained models, move to ../checkpoints/ dir
msmarco document ranking train/dev/test files
```

## Train Two Tower Document Encoder
If you want to train new document embeddings use following script:
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

To store the learned embeddings as chunks:
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
if you want to encode queries also use the flag: 
``` 
-embed_queries 1
```
The documents/queries need to be in jsonl format, per line there is one document/query as json object
```
{"doc_id": "docid", "doc": "document_content"}  
```
## Training Ranker

To train the embedding ranker the documents and the queries need to be encoded as chunks by the retriever model. 
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
The ```-document_embedding_format``` and ```-doc_ids_format``` are the formats the document embeddings and their document ids are stored respectively

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
        -loss_fn ip \
        -eval_every 5000 \
        -batch_size 32 \
        -k 1000
        -res ./results/reformulator.trec \
        -tensorboard_output ./boards/train_reformulator \
        -top_k_reformulator 10 \
        -reformulation_type [neural|weighted_avg|transformer]
```
and chose one of the ```reformulator_type``` choices and set the corresponding hyper parameters respectively. Every reformulator considers the top top_k_reformulator documents of the retrieved set.

If reformulation_type is left out, default value is None, the base architecture with knn index and ranker is evaluated.

For the neural reformulator set hidden layer dimensions, if hidden2 is 0 don't use second hidden layer:
``` 
        -hidden1 3500 \
        -hidden2 0 \
```

For the transformer reformulator set the number of attentionheads, the number of encoder layers and the dimension of the feedforward layer in each encoder layer:
``` 
        -nhead 4 \
        -num_encoder_layers 1 \
        -dim_feedforward 3072
``` 

For the weighted average there is no additionaly argument necessary. 

## Other Experiments 

With the inference.py script different experiments are possible. The most important flags are:

```
-baseline True -- evaluates the BM25 baseline with chosen anserini index 
-exact_knn True -- evaluates the Two Tower model with exact knn, big matrix multiplication of all documents and queries
-ideal -- evaluates the ideal run where the reformulated query is an actualy relevant document
-print_embeddings True -- prints the embeddings and meta data of the first 3 queries to use on projector.tensorflow.org 
-test_clustering True -- clustering method to improve recall is tested, need to set -sampling to [cluster_kmeans|cluster_spectral|attention] 
```
There are some other options which can be explored like:

``` 
-reformulate_before_ranking -- used to reformulate the query before the initial retrieved list is reranked
-use_ranker_in_next_round -- after reformulation use ranker 
-rerank_to_new_qs -- rerank the new retrieved documents after reformulation to the reformulated queries 
-avg_new_qs_for_ranking -- after reformulation average the new queries with the original ones
```

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

