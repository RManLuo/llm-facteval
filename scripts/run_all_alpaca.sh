#!/bin/bash

ROOT_DIR=/nfsdata/data/trang/chatgpt-kg
SRC_DIR=$ROOT_DIR/llm-facteval
KG=$1
GEN_TYPE=$2
MODEL=$3
HF_PIPELINE=$4
PROMPT_TEMPLATE=$5
QTYPE_IDX=$6
MODEL_NAME=$7
PROMPT_CONTEXT=${8:-none}

QTYPES=( binary_answer multiple_answer single_answer multi_choices)
ANS_SAMPLERS=( true_false short_answer short_answer 'mcq-qa' )

qtype=${QTYPES[$QTYPE_IDX]}
sampler=${ANS_SAMPLERS[$QTYPE_IDX]}
echo "======================================"
echo "Process ${qtype}"
echo "======================================"

for input_file in $ROOT_DIR/eval-data/$KG/questions_${GEN_TYPE}/*.$qtype.jsonl; do
  echo " * Processing $input_file"
  TEST=`basename $input_file .jsonl`
  ./run_answer_alpaca.sh $KG $GEN_TYPE $TEST $sampler $MODEL $HF_PIPELINE $PROMPT_TEMPLATE $MODEL_NAME $PROMPT_CONTEXT
done