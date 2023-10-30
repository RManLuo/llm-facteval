#!/bin/bash

git clone https://github.com/tatsu-lab/stanford_alpaca.git
cd stanford_alpaca

LLAMA_DIR=$ROOT_DIR/models/llama-hf/7B
ALPACA_DELTA_DIR=$ROOT_DIR/alpaca-7b-wdiff
ALPACA_DIR=$ROOT_DIR/models/alpaca-hf/7B
python weight_diff.py recover --path_raw $LLAMA_DIR --path_diff $ALPACA_DELTA_DIR  --path_tuned $ALPACA_DIR