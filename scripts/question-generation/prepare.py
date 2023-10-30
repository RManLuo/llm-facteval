import argparse
import glob
import json
import logging
import os
import random
logger = logging.getLogger()

def load_relation(relation_file, relation_name):
    data = []
    with open(relation_file, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    object_nodes = {
    }
    subject_nodes = {
    }
    records = {}
    for sample in data:
        subject_label = sample["sub_label"]  # X
        object_label = sample["obj_label"]  # Y
        object_uri = sample["obj_uri"]
        subject_uri = sample["sub_uri"]
        key = f"{subject_uri}:{object_uri}:{relation_name}"
        records[key] = sample
        if object_uri not in object_nodes:
            object_nodes[object_uri] = {
                "object_label": object_label,
                "object_uri": object_uri,
                "subjects": []
            }
        if subject_uri not in subject_nodes:
            subject_nodes[subject_uri] = {
                "subject_label": subject_label,
                "subject_uri": subject_uri,
                "objects": []
            }
        object_nodes[object_uri]["subjects"].append({
            "subject_label": subject_label,
            "subject_uri": subject_uri,
        })
        subject_nodes[subject_uri]["objects"].append({
            "object_label": object_label,
            "object_uri": object_uri,
        })

    return records, subject_nodes, object_nodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='Data dir')
    parser.add_argument('--relation-dir', type=str, help='Relation data dir')
    parser.add_argument('--out-dir', type=str, help='Out dir', default="test")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_questions = sorted(glob.glob(f"{args.data_dir}/*.jsonl"))
    for question_file in all_questions:
        logger.info(f"Process {question_file}")
        basename = os.path.splitext(os.path.basename(question_file))[0]
        relation_file = f"{args.relation_dir}/{basename}.jsonl"
        records, subject_nodes, object_nodes = load_relation(relation_file, basename)
        with open(f"{args.out_dir}/{basename}.jsonl", 'w') as fout:
            for key, record in records.items():
                evidences = record['evidences']
                for evidences in evidences:
                    quest = {
                        'question_id': key,
                        'question_type': 'single_answer',
                        'answers': [{
                            'uri': record['obj_uri'],
                            'label': record['obj_label'],
                        }],
                        'lm_input': {
                            'message': evidences['masked_sentence'],
                            'prompt': ''
                        }
                    }
                    json_record = json.dumps(quest, ensure_ascii=False)
                    fout.write(f"{json_record}\n")
