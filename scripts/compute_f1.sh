#!/bin/bash


ROOT_DIR=eval-result  # TODO: path to results folder
cd scripts/evaluation
##################
# Eval results True-False Question
##################
for KG in google_re trex wiki_bio umls; do
  DATA_DIR=$ROOT_DIR/$KG
  PROMPT_CONTEXT=none
  TEMPLATE=binary_answer
  for GEN_TYPE in template masking; do
    for MODEL in 'chatgpt' 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl' 'fastchat-t5-3b' 't5-xl' 'flan-alpaca-xl'; do
      python metrics/true-false-question.py --data-dir $DATA_DIR/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT \
    --out-file result.$TEMPLATE.txt
    done
  done
done

KG=google_re_with_context
DATA_DIR=$ROOT_DIR/$KG
TEMPLATE=binary_answer
for PROMPT_CONTEXT in relevance irrelevance antifactual; do
  for GEN_TYPE in template masking; do
    for MODEL in 'chatgpt' 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl'  'fastchat-t5-3b'  't5-xl' 'flan-alpaca-xl'; do
      python metrics/true-false-question.py --data-dir $DATA_DIR/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT \
    --out-file result.$TEMPLATE.txt
    done
  done
done

#####################
# MCQ
#####################
## all
for KG in google_re trex wiki_bio umls; do
  DATA_DIR=$ROOT_DIR/$KG
  PROMPT_CONTEXT=none
  TEMPLATE=multi_choices
  for GEN_TYPE in template masking qa; do
    for MODEL in 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl'  'fastchat-t5-3b' 't5-xl' 'flan-alpaca-xl'; do
      python metrics/eval-mcq-answer.py --data-dir $DATA_DIR/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT \
    --out-file result.$TEMPLATE.txt --delim '-'
    done

    ## chatgpt
    MODEL=chatgpt
    python metrics/eval-mcq-answer.py --data-dir $DATA_DIR/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT \
    --out-file result.$TEMPLATE.txt --delim ','
  done
done

KG=google_re_with_context
DATA_DIR=$ROOT_DIR/$KG
for PROMPT_CONTEXT in relevance irrelevance antifactual; do
  TEMPLATE=multi_choices
  for GEN_TYPE in template masking qa; do
    for MODEL in 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl'  'fastchat-t5-3b'  't5-xl' 'flan-alpaca-xl'; do
      python metrics/eval-mcq-answer.py --data-dir $DATA_DIR/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT \
    --out-file result.$TEMPLATE.txt --delim '-'
    done

    ## chatgpt
    MODEL=chatgpt
    python metrics/eval-mcq-answer.py --data-dir $DATA_DIR/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT \
    --out-file result.$TEMPLATE.txt --delim ','
  done
done

################
# FiB and SAQ
################

# Google_RE
RELATION_DIR=kg_examples/Google_RE
TEMPLATE=single_answer
KG=google_re
PROMPT_CONTEXT=none
SCRIPT_NAME=fuzzy_match_google_re.py
for GEN_TYPE in template masking qa; do
  for MODEL in 'chatgpt' 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl'  'fastchat-t5-3b'; do
    python3 metrics/$SCRIPT_NAME --data-dir $ROOT_DIR/$KG/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT --relation-dir $RELATION_DIR \
    --template $TEMPLATE --out-file result.$TEMPLATE.txt
  done
done

## google re with context
RELATION_DIR=kg_examples/Google_RE
TEMPLATE=single_answer
KG=google_re_with_context
PROMPT_CONTEXT=none
SCRIPT_NAME=fuzzy_match_google_re.py
for PROMPT_CONTEXT in relevance irrelevance antifactual; do
  for GEN_TYPE in template masking qa; do
    for MODEL in 'chatgpt' 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl'  'fastchat-t5-3b'; do
      python3 metrics/$SCRIPT_NAME --data-dir $ROOT_DIR/$KG/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT --relation-dir $RELATION_DIR \
      --template $TEMPLATE --out-file result.$TEMPLATE.txt
    done
  done
done

## trex
RELATION_DIR=kg_examples/TREx
TEMPLATE=single_answer
KG=trex
PROMPT_CONTEXT=none
SCRIPT_NAME=fuzzy_match_trex.py
for TEMPLATE in single_answer multiple_answer; do
  for GEN_TYPE in template masking qa; do
    for MODEL in 'chatgpt' 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl'  'fastchat-t5-3b'    't5-xl' 'flan-alpaca-xl'; do
      python3 metrics/$SCRIPT_NAME --data-dir $ROOT_DIR/$KG/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT --relation-dir $RELATION_DIR \
      --template $TEMPLATE --out-file result.$TEMPLATE.txt
    done
  done
done

## wikibio
RELATION_DIR=kg_examples/wiki-bio/triples_processed
KG=wiki_bio
PROMPT_CONTEXT=none
SCRIPT_NAME=fuzzy_match_wikibio.py
for TEMPLATE in single_answer multiple_answer; do
  for GEN_TYPE in template masking qa; do
    for MODEL in 'chatgpt' 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl'  'fastchat-t5-3b'  't5-xl' 'flan-alpaca-xl'; do
      python3 metrics/$SCRIPT_NAME --data-dir $ROOT_DIR/$KG/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT --relation-dir $RELATION_DIR \
      --template $TEMPLATE --out-file result.$TEMPLATE.txt
    done
  done
done

## umls
RELATION_DIR=kg_examples/umls/triples_processed
KG=umls
PROMPT_CONTEXT=none
SCRIPT_NAME=fuzzy_match_wikibio.py
for TEMPLATE in single_answer multiple_answer; do
  for GEN_TYPE in template masking qa; do
    for MODEL in 'chatgpt' 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl'  'fastchat-t5-3b'  't5-xl' 'flan-alpaca-xl'; do
      python3 metrics/$SCRIPT_NAME --data-dir $ROOT_DIR/$KG/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT --relation-dir $RELATION_DIR \
      --template $TEMPLATE --out-file result.$TEMPLATE.txt
    done
  done
done


################
# collect result
################
for KG in google_re trex wiki_bio google_re_with_context umls; do
  DATA_DIR=eval-result/$KG
  OUTPUT=$DATA_DIR/result
  python collect-main-result-single-answer.py --data-dir $DATA_DIR --out-file $OUTPUT
done

### collect_all result
OUT_DIR=combined-result
rm -rf $OUT_DIR
mkdir -p $OUT_DIR

for KG in google_re trex wiki_bio google_re_with_context umls; do
  mkdir -p $OUT_DIR/$KG
  cp $DATA_DIR/$KG/*.csv $OUT_DIR/$KG
done
