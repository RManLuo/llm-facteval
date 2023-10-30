import argparse
import glob
import json
import os
import re
import string
def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.split("\n")[0]
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

def get_choices(message, delim=','):
    full_question = message.strip().lower().split('\n\n')
    for msg in full_question:
        if msg.startswith('choices:'):
            choices = msg.replace('choices:', '').strip().split(delim)
            return [x.strip() for x in choices if x]
    return []

def test_unknown_response(unk_list, response):
    normalize_response = response.lower()
    for unk in unk_list:
        if unk in normalize_response:
            return True
    return False

def check_result(s, choices, answer):
    s = normalize(s)
    answer = normalize(answer)
    num = 0
    for c in choices:
        num += int(normalize(c) in s)
    correct = (answer in s) and (num == 1)
    return correct

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

def eval_mcq_question(data_dir, fout, fdetail, unknown_responses, delim=',', template='multi_choices'):
    all_outputs = sorted(glob.glob(f"{data_dir}/output.*.{template}.jsonl"))
    all_total = 0
    all_correct = 0
    all_incorrect = 0
    all_irrelevance = 0
    all_abstain = 0
    label_map = {
        'correct': 0,
        'incorrect': 1,
    }
    for pred_file in all_outputs:
        correct = 0
        incorrect = 0
        irrelevance = 0
        abstain = 0
        total = 0
        print(f"...... Processing {pred_file}")
        basename = os.path.splitext(os.path.basename(pred_file))[0]
        relation_name = basename.split('.')[1]
        eval_file = f"{data_dir}/{basename}.eval.txt"
        with open(pred_file) as fin, open(eval_file, 'w') as feval:
            for line in fin:
                data = json.loads(line.strip())
                total += 1
                predict = data['predicted'].strip().lower()
                if test_unknown_response(unknown_responses, predict):
                    abstain += 1
                    feval.write(f"{-1}\n")
                    continue
                choices = get_choices(data['lm_input']['message'], delim)
                is_correct = check_result(predict, choices, data['ground_truth'][0]['label'])
                feval.write(f"{int(is_correct)}\n")
                correct += int(is_correct)
                incorrect += int(1-is_correct)

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
        fout.write(f"[MCQ Questions] F1 = {f1 * 100:.2f} ; "
                   f"Irrelevant prediction = {all_irrelevance}/{all_total};"
                   f"Abstention = {all_abstain}/{all_total}\n")
        fdetail.write(
            f"all,{all_total},{all_correct}, {all_incorrect},{all_abstain},{all_irrelevance},"
            f"{precision:.2f},{recall:.2f},{f1:.2f}, {invalid_ratio:.2f}\n")
    fout.write(f"=============================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, help='Data dir',
                        default='/home/trang/workspace/projects/2023-chatgpt-kg/flan-t5-none')
    parser.add_argument('--out-file', type=str, help='Out file', default="result.txt")
    parser.add_argument('--delim', type=str, default='-', help='Choice delim')
    args = parser.parse_args()

    fout = open(f"{args.data_dir}/{args.out_file}", 'w')
    fdetail = open(f"{args.data_dir}/{args.out_file}.detail.csv", 'w')
    print(args)
    print("...Process mcq question results")
    fdetail.write(f"relation,size,correct,incorrect,abstain,invalid,precision,recall,f1, invalid_ratio\n")

    unk_file = f"{os.getcwd()}/unknown_response.txt"
    print(f"Load unknown response template from {unk_file}")
    unknown_responses = []
    with open(unk_file) as fin:
        for line in fin:
            unknown_responses.append(line.strip())

    eval_mcq_question(args.data_dir, fout, fdetail, unknown_responses, delim=args.delim)
    fout.close()
    fdetail.close()

