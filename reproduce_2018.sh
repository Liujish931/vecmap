#!/bin/bash

python3 map_embeddings.py --aaai2018 data/dictionaries/en-it.train.txt data/embeddings/original/en.emb.txt data/embeddings/original/it.emb.txt data/embeddings/en.txt data/embeddings/it.txt

python3 eval_translation.py data/embeddings/en.txt data/embeddings/it.txt -d data/dictionaries/en-it.test.txt