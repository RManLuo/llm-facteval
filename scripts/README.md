# Reproduction of EMNLP2023 Experiments
## Question Generation
### Extract triplets and generate questions

### Preprocessing questions

## Answer Prompting
### Without context

### With context


## Evaluation and Analysis
### Compute accuracy
#### True False question (TFQ)
For TFQ, the accuracy and abstention are computed using the script [true-false-question.py](evaluation/metrics/true-false-question.py).
```bash
# Without context
for KG in google_re trex wiki_bio umls; do
    RESULT_DIR=eval-result/$KG
    PROMPT_CONTEXT=none
    TEMPLATE=binary_answer
    SRC_DIR=
    for GEN_TYPE in template masking; do
        for MODEL in 'chatgpt' 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl' 'fastchat-t5-3b' 't5-xl' 'flan-alpaca-xl'; do
            python3 $SRC_DIR/scripts/evaluation/metrics/true-false-question.py --data-dir $DATA_DIR/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT \
              --out-file result.$TEMPLATE.txt
        done
    done
done

# With context
KG=google_re_with_context
DATA_DIR=eval-result/$KG
TEMPLATE=binary_answer
for PROMPT_CONTEXT in relevance irrelevance antifactual; do
  for GEN_TYPE in template masking; do
    for MODEL in 'chatgpt' 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl'  'fastchat-t5-3b'  't5-xl' 'flan-alpaca-xl'; do
      python $SRC_DIR/scripts/evaluation/metrics/true-false-question.py --data-dir $DATA_DIR/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT \
    --out-file result.$TEMPLATE.txt
    done
  done
done
```

#### Multiple Choice Question (MCQ)
For TFQ, the accuracy and abstention are computed using the script [eval-mcq-answer.py](evaluation/metrics/eval-mcq-answer.py).
```bash
# For ChatGPT - we use ',' as delimiter
for KG in google_re trex wiki_bio umls; do
  DATA_DIR=eval-result/$KG
  PROMPT_CONTEXT=none
  TEMPLATE=multi_choices
  MODEL=chatgpt
  for GEN_TYPE in template masking qa; do
    python $SRC_DIR/scripts/evaluation/metrics/eval-mcq-answer.py --data-dir $DATA_DIR/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT \
    --out-file result.$TEMPLATE.txt --delim ','
  done
done

# Other LLMs, '-' as delimiter
for KG in google_re trex wiki_bio umls; do
  DATA_DIR=eval-result/$KG
  PROMPT_CONTEXT=none
  TEMPLATE=multi_choices
  for GEN_TYPE in template masking qa; do
    for MODEL in 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl'  'fastchat-t5-3b' 't5-xl' 'flan-alpaca-xl'; do
      python $SRC_DIR/scripts/evaluation/metrics/eval-mcq-answer.py --data-dir $DATA_DIR/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT \
    --out-file result.$TEMPLATE.txt --delim '-'
    done
  done
done
```

For with context experiments, similar to TFQ, please use `google_re_with_context` as KG and set `PROMPT_CONTEXT` with one of the values `[relevance, irrelevance, antifactual]`.

### Fill in the blank (FiB) and Short Answer Question (SAQ)
We use fuzzy_match to calculate accuracy of FiB and SAQ questions. Together with the fuzzy match, we also validate whether the answer is incorrect form, i.e. number for years, datetime, etc. Therefore, the fuzzy match evaluation is relation-dependent. Each KG has its own fuzzy match implementation in [metrics](evaluation/metrics)
```bash
# Evaluation for Google_RE
KG=google_re
DATA_DIR=eval-result/$KG
RELATION_DIR=kg_examples/Google_RE
TEMPLATE=single_answer
PROMPT_CONTEXT=none
SCRIPT_NAME=$SRC_DIR/scripts/evaluation/metrics/fuzzy_match_google_re.py
for GEN_TYPE in template masking qa; do
  for MODEL in 'chatgpt' 'alpaca-7b-fp16' 'vicuna-7b-fp16' 'llama-7b' 'flan-t5' 'flan-t5-xl'  'fastchat-t5-3b'; do
    python3 $SCRIPT_NAME --data-dir $DATA_DIR/questions_${GEN_TYPE}/$MODEL-$PROMPT_CONTEXT --relation-dir $RELATION_DIR \
    --template $TEMPLATE --out-file result.$TEMPLATE.txt
  done
done
```
## Collect results and compute F1 score
```bash
KG=google_re
DATA_DIR=eval-result/$KG
OUTPUT=$DATA_DIR/result
python collect-main-result-single-answer.py --data-dir $DATA_DIR --out-file $OUTPUT

KG=trex
DATA_DIR=eval-result/$KG
OUTPUT=$DATA_DIR/result
python collect-main-result.py --data-dir $DATA_DIR --out-file $OUTPUT

KG=wiki_bio
DATA_DIR=eval-result/$KG
OUTPUT=$DATA_DIR/result
python collect-main-result.py --data-dir $DATA_DIR --out-file $OUTPUT

KG=umls
DATA_DIR=eval-result/$KG
OUTPUT=$DATA_DIR/result
python collect-main-result.py --data-dir $DATA_DIR --out-file $OUTPUT

KG=google_re_with_context
DATA_DIR=eval-result/$KG
OUTPUT=$DATA_DIR/result
python collect-context-prompting-result.py --data-dir $DATA_DIR --out-file $OUTPUT
```

### Analysis
#### Question Diversity