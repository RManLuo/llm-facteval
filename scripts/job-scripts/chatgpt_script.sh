#!/bin/bash

ROOT_DIR=/nfsdata/data/trang/chatgpt-kg
MODEL_NAME=chatgpt
PROMPT_CONTEXT=none

## True-False
PROMPT_TEMPLATE=$ROOT_DIR/chatgptEval/conf/prompts/true-false/chatgpt.txt
for KG in google_re trex wiki_bio; do
  for GEN_TYPE in template masking; do
    for QTYPE_IDX in 0 1; do
      sbatch --job-name=$KG-$GEN_TYPE-$QTYPE_IDX-$MODEL_NAME-$PROMPT_CONTEXT  submit-job-cpu.sh \
    run_all_chatgpt.sh $KG $GEN_TYPE $OPENAI_API_KEY $PROMPT_TEMPLATE \
    $QTYPE_IDX $PROMPT_CONTEXT
    done
  done
done

## Masking
GEN_TYPE=masking
PROMPT_TEMPLATE=$ROOT_DIR/chatgptEval/conf/prompts/short-answer-masking/chatgpt.txt
for KG in google_re trex wiki_bio; do
  for QTYPE_IDX in 1 2; do
    sbatch --job-name=$KG-$GEN_TYPE-$QTYPE_IDX-$MODEL_NAME-$PROMPT_CONTEXT submit-job-cpu.sh \
    run_all_chatgpt.sh $KG $GEN_TYPE $OPENAI_API_KEY $PROMPT_TEMPLATE \
    $QTYPE_IDX $PROMPT_CONTEXT
  done
done

## short-answer
GEN_TYPE=qa
PROMPT_TEMPLATE=$ROOT_DIR/chatgptEval/conf/prompts/short-answer-qa/chatgpt.txt
for KG in google_re trex wiki_bio; do
  for QTYPE_IDX in 1 2; do
    sbatch --job-name=$KG-$GEN_TYPE-$QTYPE_IDX-$MODEL_NAME-$PROMPT_CONTEXT submit-job-cpu.sh \
    run_all_chatgpt.sh $KG $GEN_TYPE $OPENAI_API_KEY $PROMPT_TEMPLATE \
    $QTYPE_IDX $PROMPT_CONTEXT
  done
done

## mcq
QTYPE_IDX=3
GEN_TYPE=masking
PROMPT_TEMPLATE=$ROOT_DIR/chatgptEval/conf/prompts/mcq-masking/chatgpt.txt
for KG in google_re trex wiki_bio; do
  sbatch --job-name=$KG-$GEN_TYPE-$QTYPE_IDX-$MODEL_NAME submit-job-cpu.sh \
  run_all_chatgpt.sh $KG $GEN_TYPE $OPENAI_API_KEY $PROMPT_TEMPLATE \
  $QTYPE_IDX $PROMPT_CONTEXT
done

GEN_TYPE=qa
QTYPE_IDX=3
for KG in google_re trex wiki_bio; do
  PROMPT_TEMPLATE=$ROOT_DIR/chatgptEval/conf/prompts/mcq-qa/chatgpt.txt
    sbatch --job-name=$KG-$GEN_TYPE-$QTYPE_IDX-$MODEL_NAME submit-job-cpu.sh \
    run_all_chatgpt.sh $KG $GEN_TYPE $OPENAI_API_KEY $PROMPT_TEMPLATE \
    $QTYPE_IDX $PROMPT_CONTEXT
done
