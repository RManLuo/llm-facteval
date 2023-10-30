import argparse
import glob
import json
import os
from dateutil.parser import parse
from nltk import word_tokenize
from sklearn.metrics import accuracy_score
import copy
import re
import string
from collections import Counter, defaultdict


def get_answer(text, answer_prompt):
    idx = text.rfind(answer_prompt)
    if idx == -1:
        return None
    return text[idx + len(answer_prompt) :]


def get_consensus(answers):
    counts = defaultdict(int)
    for answer in answers:
        counts[answer] += 1
    counts[None] = 0
    return max(counts, key=counts.get)


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.split("\n")[0]
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1


def get_scores_from_text(text: str) -> dict:
    pattern = r"## (.+?)\n.+?(\d)/5"
    matches = re.findall(pattern, text, re.DOTALL)
    return {k: int(v) for k, v in dict(matches).items()}


def get_yesno_from_text(text: str) -> dict:
    pattern = r"## (.+?)\n.+?([yn])"
    matches = re.findall(pattern, text, re.DOTALL)
    return {k: v for k, v in dict(matches).items()}


def get_letter_from_data(data: str) -> str:
    last_y = (data.rfind("y"), "y")
    last_n = (data.rfind("n"), "n")
    char = max(last_y, last_n)[1]
    return char


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

def test_unknown_response(unk_list, response):
    normalize_response = response.lower()
    for unk in unk_list:
        if unk in normalize_response:
            return True
    return False

def calculate_f1(correct, incorrect, abstain):
    if (correct + incorrect) == 0:
        precision = 0
    else:
        precision = float(correct / (correct + incorrect)) * 100
    if (correct + incorrect + abstain) == 0:
        recall = 0
    else:
        recall = float((correct) / (correct + incorrect + abstain)) * 100
    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * float((precision * recall) / (precision + recall))
    return precision, recall, f1_score

def eval_short_answer_question(data_dir, fout, fdetail, relation_dir, unknown_responses, template='single_answer'):
    all_total = 0
    all_correct = 0
    all_incorrect = 0
    all_irrelevance = 0
    all_abstain = 0
    all_outputs = sorted(glob.glob(f"{args.data_dir}/output.*.{template}.jsonl"))
    for pred_file in all_outputs:
        correct = 0
        incorrect = 0
        irrelevance = 0
        abstain = 0
        total = 0
        basename = os.path.splitext(os.path.basename(pred_file))[0]
        relation_name = basename.split('.')[1]
        relation_file = f"{args.relation_dir}/{relation_name}.jsonl"
        records, subject_nodes, object_nodes = load_relation(relation_file, relation_name)
        eval_file = f"{data_dir}/{basename}.eval.txt"
        with open(pred_file) as fin, open(eval_file, 'w') as feval:
            for line in fin:
                data = json.loads(line.strip())
                total += 1
                # ans = data['ground_truth'][0]['label']
                question_id = data['question_id'].split(':')
                sub_id = question_id[0]
                obj_id = question_id[1]
                rel_id = question_id[2]
                record_key = ':'.join(question_id[:3])
                record = records[record_key]
                predict = data['predicted'].strip()
                if test_unknown_response(unknown_responses, predict):
                    abstain += 1
                    feval.write(f"{-1}\n")
                    continue
                is_correct = False
                for correct_ans in data['ground_truth']:
                    is_correct = is_correct or fuzzy_match(correct_ans['label'], predict)
                feval.write(f"{int(is_correct)}\n")
                correct += int(is_correct)
                incorrect += (1 - int(is_correct))
            assert total == (correct + incorrect + abstain)
            if total > 0:
                precision, recall, f1 = calculate_f1(correct, incorrect, abstain)
                invalid_ratio = float(irrelevance / total) * 100
                fout.write(f"[{os.path.basename(pred_file)}] F1 = {f1:.2f} ; "
                           f"Irrelevant prediction = {irrelevance}/{total};"
                           f"Abstention = {abstain}/{total}\n")
                fdetail.write(
                    f"{os.path.basename(pred_file).split('.')[1]},{total},{correct}, {incorrect},{abstain},{irrelevance},"
                    f"{precision:.2f},{recall:.2f},{f1:.2f}, {invalid_ratio:.2f}\n")
            all_irrelevance += irrelevance
            all_total += total
            all_correct += correct
            all_incorrect += incorrect
            all_abstain += abstain
    assert all_total == (all_correct + all_incorrect + all_abstain)
    fout.write(f"=============================================\n")
    if all_total > 0:
        precision, recall, f1 = calculate_f1(all_correct, all_incorrect, all_abstain)
        invalid_ratio = float(all_irrelevance / all_total) * 100
        fout.write(f"[Fuzzy match] F1 = {f1 * 100:.2f} ; "
                   f"Irrelevant prediction = {all_irrelevance}/{all_total};"
                   f"Abstention = {all_abstain}/{all_total}\n")
        fdetail.write(
            f"all,{all_total},{all_correct}, {all_incorrect},{all_abstain},{all_irrelevance},"
            f"{precision:.2f},{recall:.2f},{f1:.2f}, {invalid_ratio:.2f}\n")
    fout.write(f"=============================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, help='Data dir', default='/home/tvuu0014/workspace/certfiedlm/eval-result/trex/questions_template/chatgpt')
    parser.add_argument('--out-file', type=str, help='Out file', default="result.txt")
    parser.add_argument('--relation-dir', type=str, help='Relation dir',
                        default='/home/tvuu0014/workspace/certfiedlm/chatgptEval/kg_examples/TREx')
    parser.add_argument('--template', type=str, help='template',
                        default='single_answer')
    args = parser.parse_args()

    fout = open(f"{args.data_dir}/{args.out_file}", 'w')
    print(args)
    fdetail = open(f"{args.data_dir}/{args.out_file}.detail.csv", 'w')
    fdetail.write(f"relation,size,correct,incorrect,abstain,invalid,precision,recall,f1, invalid_ratio\n")

    unk_file = f"{os.getcwd()}/unknown_response.txt"
    print(f"Load unknown response template from {unk_file}")
    unknown_responses = []
    with open(unk_file) as fin:
        for line in fin:
            unknown_responses.append(line.strip())

    print("...Process short answer question results")
    eval_short_answer_question(args.data_dir, fout, fdetail, args.relation_dir, unknown_responses, template=args.template)
    fout.close()
    fdetail.close()

