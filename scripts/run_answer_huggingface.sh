#!/bin/bash

ROOT_DIR=/nfsdata/data/trang/chatgpt-kg
SRC_DIR=$ROOT_DIR/llm-facteval
KG=$1
GEN_TYPE=$2
TEST=$3
ANS_SAMPLER=$4
MODEL=$5
HF_PIPELINE=$6
PROMPT_TEMPLATE=$7
MODEL_NAME=$8
PROMPT_CONTEXT=$9

export TMPDIR=$ROOT_DIR/tmp
export HF_HOME=$ROOT_DIR/huggingface
export HUGGINGFACE_HUB_CACHE=$ROOT_DIR/huggingface
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1

if [ "$#" -lt 6 ]; then
    echo "Illegal number of parameters"
    echo Usage "$0 <KG> <Question_gen_type> <test file> <answer sampler> <HF model> <HF pipeline"
    exit 1
fi


INPUT_FILE=$ROOT_DIR/eval-data/$KG/questions_${GEN_TYPE}/${TEST}.jsonl
OUT_DIR=$ROOT_DIR/eval-result/$KG/questions_${GEN_TYPE}/$MODEL_NAME
mkdir -p $OUT_DIR

OUT_FILE=$OUT_DIR/output.${TEST}.jsonl

if [ -f "${OUT_FILE}" ]
then
    echo "Exp completed. Skip."
    exit
fi

EVAL_LM=huggingface_plm
module load anaconda/anaconda3
source activate $ROOT_DIR/env

echo "=================================================="
echo " Running answer generation"
echo " answer-sampler : "$ANS_SAMPLER
echo " input-file     : "$INPUT_FILE
echo " output-file    : "$OUT_FILE
echo " eval-lm        : "$EVAL_LM
echo " model          : "$MODEL
echo " hf-pipeline    : "$HF_PIPELINE
echo " prompt-template: "$PROMPT_TEMPLATE
echo " prompt-context : "$PROMPT_CONTEXT
echo "=================================================="

python3 $SRC_DIR/run_certlm.py --step answer_sampling \
   --answer-sampler $ANS_SAMPLER \
   --input-file $INPUT_FILE \
   --output-file $OUT_FILE \
   --eval-lm $EVAL_LM \
   --model $MODEL --hf-pipeline $HF_PIPELINE --device 0 \
   --prompt-template $PROMPT_TEMPLATE --prompt-context $PROMPT_CONTEXT
