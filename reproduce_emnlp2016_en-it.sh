#!/bin/bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA="$ROOT/data"
OUTPUT="$ROOT/output/emnlp2016"
TRAIN_DICTIONARY="$DATA/dictionaries/en-it.train.txt"
TEST_DICTIONARY="$DATA/dictionaries/en-it.test.txt"
TEST_ANALOGIES="$DATA/analogies/questions-words.txt"




python3 "$ROOT/eval_translation.py" "$OUTPUT/orthogonal-unit-center/en.emb.txt" "$OUTPUT/orthogonal-unit-center/it.emb.txt" -d "$TEST_DICTIONARY"
