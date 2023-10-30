import argparse
import glob
import json
import os
from dateutil.parser import parse
from nltk import word_tokenize
from sklearn.metrics import accuracy_score

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

def eval_true_false_question(data_dir, fout, fdetail, unknown_responses, template='binary_answer'):
    all_outputs = sorted(glob.glob(f"{args.data_dir}/output.*.{template}.jsonl"))
    all_correct = 0
    all_incorrect = 0
    all_irrelevance = 0
    all_abstain = 0
    all_total = 0
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
        eval_file = f"{data_dir}/{basename}.eval.txt"
        with open(pred_file) as fin, open(eval_file, 'w') as feval:
            for line in fin:
                total += 1
                data = json.loads(line.strip())
                predict = data['predicted'].strip().lower()
                if test_unknown_response(unknown_responses, predict):
                    abstain += 1
                    feval.write(f"{-1}\n")
                    continue
                predict = word_tokenize(predict)
                if len(predict) == 1:
                    if predict[0] in label_map:
                        if label_map[data['ground_truth']] == label_map[predict[0]]:
                            correct += 1
                        else:
                            incorrect += 1
                        feval.write(f"{int(label_map[data['ground_truth']] == label_map[predict[0]])}\n")
                    else:
                        irrelevance += 1
                        incorrect += 1
                        feval.write(f"0\n")
                else:
                    predict_set = set(predict)
                    found = False
                    for key in label_map.keys():
                        if key in predict_set:
                            if label_map[data['ground_truth']] == label_map[key]:
                                correct += 1
                            else:
                                incorrect += 1
                            feval.write(f"{int(label_map[data['ground_truth']] == label_map[key])}\n")
                            found = True
                            break
                    if not found:
                        incorrect += 1
                        irrelevance += 1
                        feval.write(f"0\n")
            assert total == (correct + incorrect + abstain)
            if total > 0:
                precision, recall, f1 = calculate_f1(correct, incorrect, abstain)
                invalid_ratio = float(irrelevance / total) * 100
                fout.write(f"[{os.path.basename(pred_file)}] F1 = {f1:.2f} ; "
                           f"Irrelevant prediction = {irrelevance}/{total};"
                           f"Abstention = {abstain}/{total}\n")
                fdetail.write(f"{os.path.basename(pred_file).split('.')[1]},{total},{correct}, {incorrect},{abstain},{irrelevance},"
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
        fout.write(f"[True-False Questions] F1 = {f1 * 100:.2f} ; "
                   f"Irrelevant prediction = {all_irrelevance}/{all_total};"
                   f"Abstention = {all_abstain}/{all_total}\n")
        fdetail.write(
            f"all,{all_total},{all_correct}, {all_incorrect},{all_abstain},{all_irrelevance},"
            f"{precision:.2f},{recall:.2f},{f1:.2f}, {invalid_ratio:.2f}\n")
    fout.write(f"=============================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, help='Data dir')
    parser.add_argument('--out-file', type=str, help='Out file', default="result.txt")
    args = parser.parse_args()

    fout = open(f"{args.data_dir}/{args.out_file}", 'w')
    fdetail = open(f"{args.data_dir}/{args.out_file}.detail.csv", 'w')
    print(args)
    print("...Process true-false question results")
    fdetail.write(f"relation,size,correct,incorrect,abstain,invalid,precision,recall,f1, invalid_ratio\n")

    unk_file = f"{os.getcwd()}/unknown_response.txt"
    print(f"Load unknown response template from {unk_file}")
    unknown_responses = []
    with open(unk_file) as fin:
        for line in fin:
            unknown_responses.append(line.strip())

    eval_true_false_question(args.data_dir, fout, fdetail, unknown_responses)
    fout.close()
    fdetail.close()

