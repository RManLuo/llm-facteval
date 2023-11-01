#!/bin/bash

ROOT_DIR=/nfsdata/data/trang/chatgpt-kg
MODEL='huggyllama/llama-7b'
HF_PIPELINE=text-generation

PROMPT_CONTEXT=none
MODEL_NAME=llama-7b-$PROMPT_CONTEXT
TEMPLATE_NAME=alpaca

##### TRUE-FALSE
GEN_TYPE=template
PROMPT_TEMPLATE=$ROOT_DIR/chatgptEval/conf/prompts/true-false/$TEMPLATE_NAME.txt
QTYPE_IDX=0
for KG in google_re trex wiki_bio; do
  for GEN_TYPE in template masking; do
    sbatch --job-name=$KG-$GEN_TYPE-$QTYPE_IDX-$MODEL_NAME --exclude="node[01-05]" submit-job-1gpu.sh \
    run_all_huggingface.sh $KG $GEN_TYPE $MODEL $HF_PIPELINE $PROMPT_TEMPLATE \
    $QTYPE_IDX $MODEL_NAME $PROMPT_CONTEXT
  done
done

##### Cloze test - masking
GEN_TYPE=masking
PROMPT_TEMPLATE=$ROOT_DIR/chatgptEval/conf/prompts/short-answer-masking/$TEMPLATE_NAME.txt
for QTYPE_IDX in 1 2; do
  for KG in google_re trex wiki_bio; do
    sbatch --job-name=$KG-$GEN_TYPE-$QTYPE_IDX-$MODEL_NAME --exclude="node[01-05]" submit-job-1gpu.sh \
    run_all_huggingface.sh $KG $GEN_TYPE $MODEL $HF_PIPELINE $PROMPT_TEMPLATE \
    $QTYPE_IDX $MODEL_NAME $PROMPT_CONTEXT
  done
done

#### short answer
GEN_TYPE=qa
PROMPT_TEMPLATE=$ROOT_DIR/chatgptEval/conf/prompts/short-answer-qa/$TEMPLATE_NAME.txt
for QTYPE_IDX in 1 2; do
  for KG in google_re trex wiki_bio; do
    sbatch --job-name=$KG-$GEN_TYPE-$QTYPE_IDX-$MODEL_NAME --exclude="node[01-05]" submit-job-1gpu.sh \
    run_all_huggingface.sh $KG $GEN_TYPE $MODEL $HF_PIPELINE $PROMPT_TEMPLATE \
    $QTYPE_IDX $MODEL_NAME $PROMPT_CONTEXT
  done
done


##### mcq
QTYPE_IDX=3
for KG in google_re trex wiki_bio; do
  for GEN_TYPE in template masking; do
    PROMPT_TEMPLATE=$ROOT_DIR/chatgptEval/conf/prompts/mcq-$GEN_TYPE/$TEMPLATE_NAME.txt
    sbatch --job-name=$KG-$GEN_TYPE-$QTYPE_IDX-$MODEL_NAME --exclude="node[01-05]" submit-job-1gpu.sh \
    run_all_huggingface.sh $KG $GEN_TYPE $MODEL $HF_PIPELINE $PROMPT_TEMPLATE \
    $QTYPE_IDX $MODEL_NAME $PROMPT_CONTEXT
  done
done