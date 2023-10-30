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
        object_uri = sample["obj"]
        subject_uri = sample["sub"]
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

def add_context(records, subject_nodes, object_nodes, question):
    question_id = question['question_id']
    tmp = question_id.split(':')
    sub_uri = tmp[0]
    obj_uri = tmp[1]
    relation_name = tmp[2]
    sub_info = subject_nodes[sub_uri]
    obj_info = object_nodes[obj_uri]
    record_id = f"{sub_uri}:{obj_uri}:{relation_name}"
    record = records[record_id]
    evidences = record['evidences']
    contexts = {
        'relevance': evidences[0]
    }
    subject_keys = list(subject_nodes.keys())
    true_obj = set([ele['object_uri'] for ele in sub_info['objects']])
    while True:
        idx = random.randint(0, len(subject_keys)-1)
        neg_sub = subject_keys[idx]
        if neg_sub == sub_uri:
            continue
        neg_obj = set([ele['object_uri'] for ele in subject_nodes[neg_sub]['objects']])
        obj_cand = neg_obj.difference(true_obj)
        if len(obj_cand) == 0:
            continue
        obj_id = list(obj_cand)[0]
        neg_record_id = f"{neg_sub}:{obj_id}:{relation_name}"
        contexts['irrelevance'] = records[neg_record_id]['evidences'][0]
        contexts['irrelevance']['neg_obj_id'] = obj_id
        neg_obj_label = object_nodes[obj_id]['object_label']
        anti_factual_snippet = contexts['relevance']['snippet']
        for true_label in sub_info['objects']:
            anti_factual_snippet = anti_factual_snippet.replace(true_label['object_label'], neg_obj_label)
        contexts['antifactual'] = {
            'snippet': anti_factual_snippet,
            'neg_obj_id': obj_id
        }
        break
    question['contexts'] = contexts
    return question

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
        relation_file = f"{args.relation_dir}/{basename}_test.jsonl"
        records, subject_nodes, object_nodes = load_relation(relation_file, basename)
        with open(question_file, 'r') as fin, open(f"{args.out_dir}/{basename}.jsonl", 'w') as fout:
            for line in fin:
                quest = json.loads(line.strip())
                quest_with_context = add_context(records, subject_nodes, object_nodes, quest)
                json_record = json.dumps(quest_with_context, ensure_ascii=False)
                fout.write(f"{json_record}\n")
