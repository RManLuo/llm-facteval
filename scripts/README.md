# Reproduction of EMNLP2023 Experiments
## Question Generation

### Download data
You can download the example KGs from [here](https://drive.google.com/drive/folders/1Y17Hcnh9bjmOrxfU-R2ktgu444QQh9_2?usp=sharing).

Download the `kg_examples` and put it in the root folder of the project.

### Extract triplets and generate questions
To extract triplets form the KGs, you can run the `extract_kg.sh` script. The extracted triplets will be stored in `./extracted_output` folder.
```bash
#!/bin/bash

KG_LIST="trex google_re umls wiki_bio"

for KG_NAME in ${KG_LIST}
do
    python run_certlm.py --step extract \
        --kg ${KG_NAME} \
        --data-dir ./kg_examples/${KG_NAME} \
        --output-dir ./extracted_output/${KG_NAME}/ 
done
```
After extracting the triplets, you can generate questions by using the `question-generation/generate_questions.sh` script. The generated questions will be stored in `./generated_question` folder.
```bash
./question-generation/generate_questions.sh {KG_NAME} {GENERATOR}
```
We support four KG datasets: `trex, google_re, umls, wiki_bio`.

We support three types of question generator:
- `template`: generate questions from predefined templates
- `masking`: generate questions with ChatGPT by masking the subject and object.
- `qa`: generate questions with ChatGPT with Wh-question form.

Example:
```bash
./generate_questions.sh trex template
./generate_questions.sh trex masking
./generate_questions.sh trex qa
```


### Benchmark Creation
We create the benchmark by splitting the questions by the relation name, question types and answer types. We support following question types:
- template: FiB or True/False question is generated by a predefined template. This template is written by human.
- masking: FiB question is generated by ChatGPT
- qa: Wh-question form

Answer types include
- binary_answer: True False question
- multi_choices: multiple choice questions. We provide four choices
- single_answer: short answer question with a unique correct answer, corresponding to 1-1 relation in KG
- multiple_answer: short answer question with multiple correct answers, corresponding to 1-N and M-N relation in KG.

Below is the script used to generate the benchmark dataset. Preprocessed benchmark can be found [here](https://drive.google.com/drive/folders/1anPyTxuqOa-nUZTpx_St9ukZnEZLtAHC?usp=drive_link).

```bash
DATA_DIR=generated_question
OUT_DIR=eval-data
for kg in 'google_re' 'trex' 'umls' 'wiki_bio'; do
  for qtype in 'masking' 'qa' 'template'; do
    python3 prepare.py --data-dir $DATA_DIR/$kg/questions_$qtype \
    --out-dir $OUT_DIR/$kg/questions_$qtype
  done
done
```

## Answer Prompting
The script to evaluate each LLMs can be found in [job-scripts folder](job-scripts).

### ChatGPT
```bash
KG=google_re
GEN_TYPE=template
TEST=<relation_name>
ANS_SAMPLER=true_false
OPEN_AI_KEY=
PROMPT_TEMPLATE=conf/prompts/true-false/chatgpt.txt
PROMPT_CONTEXT=none
MODEL=gpt-3.5-turbo
EVAL_LM=chat_gpt
python3 $SRC_DIR/run_certlm.py --step answer_sampling \
   --answer-sampler $ANS_SAMPLER \
   --input-file $INPUT_FILE \
   --output-file $OUT_FILE \
   --eval-lm $EVAL_LM \
   --model $MODEL  --open-ai-key $OPEN_AI_KEY \
   --prompt-template $PROMPT_TEMPLATE --prompt-context $PROMPT_CONTEXT
```

### HuggingFace model
```bash
KG=google_re
GEN_TYPE=template
TEST=<relation_name>
ANS_SAMPLER=true_false
OPEN_AI_KEY=
PROMPT_TEMPLATE=
PROMPT_CONTEXT=none
EVAL_LM=huggingface_plm
python3 $SRC_DIR/run_certlm.py --step answer_sampling \
   --answer-sampler $ANS_SAMPLER \
   --input-file $INPUT_FILE \
   --output-file $OUT_FILE \
   --eval-lm $EVAL_LM \
   --model $MODEL --hf-pipeline $HF_PIPELINE --device 0 \
   --prompt-template $PROMPT_TEMPLATE --prompt-context $PROMPT_CONTEXT
```


## Evaluation and Analysis
### Evaluation
#### True False question (TFQ)
For TFQ, the accuracy and abstention are computed using the script [true-false-question.py](evaluation/metrics/true-false-question.py). The full evaluation commands can be found in [compute_f1 script](compute_f1.sh).
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
### Collect results and compute F1 score
```bash
KG=google_re
DATA_DIR=eval-result/$KG
OUTPUT=$DATA_DIR/result
python collect-main-result-single-answer.py --data-dir $DATA_DIR --out-file $OUTPUT
```

### Analysis
#### Question Diversity
To analyze the diversity of generated questions, we use the SelfBleu metric. 
```bash
./question_diversity.sh
```
### Question Similarity
To analyze the similarity between LLM-generated questions and template-generated questions, we calculate the similarity with bert score.
```bash
./question_sim.sh
```