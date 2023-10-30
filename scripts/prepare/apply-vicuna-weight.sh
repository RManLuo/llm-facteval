#!/bin/bash

git clone https://github.com/lm-sys/FastChat.git
cd FastChat
# Follow instruction to install fastchat

export HF_HOME=$ROOT_DIR/huggingface
export HUGGINGFACE_HUB_CACHE=$ROOT_DIR/huggingface
MODEL_SIZE=7
LLAMA_DIR=$ROOT_DIR/models/llama-hf/${MODEL_SIZE}B
VICUNA_DIR=$ROOT_DIR/models/vicuna-hf/${MODEL_SIZE}B
python -m fastchat.model.apply_delta \
    --base-model-path $LLAMA_DIR \
    --target-model-path $VICUNA_DIR \
    --delta-path lmsys/vicuna-${MODEL_SIZE}b-delta-v1.1