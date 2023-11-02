KG_LIST="trex google_re umls wiki_bio"

for KG_NAME in ${KG_LIST}
do
    echo ${KG_NAME}
    python question_analysis/diversity.py --data generated_question/${KG_NAME}/questions_template -p
    python question_analysis/diversity.py --data generated_question/${KG_NAME}/questions_masking -p
    python question_analysis/diversity.py --data generated_question/${KG_NAME}/questions_qa -p
done