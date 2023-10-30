#!/bin/bash

git clone https://github.com/huggingface/transformers.git
cd transformers
# TODO: follow instruction to install transformers
MODEL_SIZE=7B

LLAMA_DIR=$ROOT_DIR/models/llama
OUTPUT_DIR=$ROOT_DIR/models/llama-hf/$MODEL_SIZE
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir $LLAMA_DIR --model_size $MODEL_SIZE --output_dir $OUTPUT_DIR