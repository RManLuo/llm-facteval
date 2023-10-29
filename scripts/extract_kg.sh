#!/bin/bash

KG_LIST="trex google_re umls wiki_bio"

for KG_NAME in ${KG_LIST}
do
    python run_certlm.py --step extract \
        --kg ${KG_NAME} \
        --data-dir ./kg_examples/${KG_NAME} \
        --output-dir ./extracted_output/${KG_NAME}/ 
done