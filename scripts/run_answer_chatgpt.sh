#!/bin/bash

ROOT_DIR=/nfsdata/data/trang/chatgpt-kg
SRC_DIR=$ROOT_DIR/llm-facteval
KG=$1
GEN_TYPE=$2
TEST=$3
ANS_SAMPLER=$4
OPEN_AI_KEY=$5
PROMPT_TEMPLATE=$6
PROMPT_CONTEXT=$7

if [ "$#" -lt 5 ]; then
    echo "Illegal number of parameters"
    echo Usage "$0 <KG> <Question_gen_type> <test file> <answer sampler> <open ai key>"
    exit 1
fi

INPUT_FILE=$ROOT_DIR/eval-data/$KG/questions_${GEN_TYPE}/${TEST}.jsonl
OUT_DIR=$ROOT_DIR/eval-result/$KG/questions_${GEN_TYPE}/chatgpt-$PROMPT_CONTEXT
mkdir -p $OUT_DIR

OUT_FILE=$OUT_DIR/output.${TEST}.jsonl

if [ -f "${OUT_FILE}" ]
then
    echo "Exp completed. Skip."
    exit
fi

MODEL=gpt-3.5-turbo
EVAL_LM=chat_gpt
module load anaconda/anaconda3
source activate $ROOT_DIR/env

echo "=================================================="
echo " Running answer generation"
echo " answer-sampler : "$ANS_SAMPLER
echo " input-file     : "$INPUT_FILE
echo " output-file    : "$OUT_FILE
echo " eval-lm        : "$EVAL_LM
echo " model          : "$MODEL
echo " prompt-template: "$PROMPT_TEMPLATE
echo " prompt-context : "$PROMPT_CONTEXT
echo "=================================================="

python3 $SRC_DIR/run_certlm.py --step answer_sampling \
   --answer-sampler $ANS_SAMPLER \
   --input-file $INPUT_FILE \
   --output-file $OUT_FILE \
   --eval-lm $EVAL_LM \
   --model $MODEL  --open-ai-key $OPEN_AI_KEY \
   --prompt-template $PROMPT_TEMPLATE --prompt-context $PROMPT_CONTEXT
