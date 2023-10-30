#!/bin/bash

KG_NAME=$1
GENERATOR=$2

if [ "$GENERATOR" = "template" ]
then
    python run_certlm.py --step question_generate \
        --kg ${KG_NAME} \
        --data-dir ./kg_examples/${KG_NAME} \
        --generator ${GENERATOR} \
        --gen-input-dir ./extracted_output/${KG_NAME}/tuples \
        --output-dir ./generated_question/${KG_NAME} \
        --num-workers 5 \
        --thread 5
elif [ "$GENERATOR" = "masking" ] || [ "$GENERATOR" = "qa" ]
then
    python run_certlm.py --step question_generate \
    --kg ${KG_NAME} \
    --generator ${GENERATOR} \
    --data-dir ./kg_examples/${KG_NAME} \
    --gen-input-dir ./extracted_output/${KG_NAME}/tuples \
    --generator-lm chat_gpt --model gpt-3.5-turbo --open-ai-key ${OPENAI_API_KEY} \
    --num-workers 1 \
    --thread 10 \
    --output-dir ./generated_question/${KG_NAME}
fi
