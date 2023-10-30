import argparse
import glob
import json
import os
from dateutil.parser import parse
from nltk import word_tokenize
from sklearn.metrics import accuracy_score

from evals.elsuite import utils


def eval_true_false_question(data_dir, fout, template='binary_answer'):
    all_outputs = sorted(glob.glob(f"{args.data_dir}/output.*.{template}.jsonl"))
    all_ground_truths = []
    all_predictions = []
    all_irrelevance = 0
    label_map = {
        'correct': 0,
        'incorrect': 1,
    }
    for pred_file in all_outputs:
        ground_truths = []
        predictions = []
        irrelevance = 0
        with open(pred_file) as fin:
            for line in fin:
                data = json.loads(line.strip())
                predict = data['predicted'].strip().lower()
                if "i'm sorry" in predict or "cannot predict" in predict:
                    irrelevance += 1
                    continue
                predict = word_tokenize(predict)
                if len(predict) == 1:
                    if predict[0] in label_map:
                        predictions.append(label_map[predict[0]])
                        ground_truths.append(label_map[data['ground_truth']])
                    else:
                        irrelevance += 1
                else:
                    predict_set = set(predict)
                    found = False
                    for key in label_map.keys():
                        if key in predict_set:
                            predictions.append(label_map[key])
                            ground_truths.append(label_map[data['ground_truth']])
                            found = True
                            break
                    if not found:
                        irrelevance += 1
            acc = accuracy_score(ground_truths, predictions)

            fout.write(f"[{os.path.basename(pred_file)}] Accuracy = {acc:.2f} ; "
                       f"Irrelevant prediction = {irrelevance}/{len(ground_truths) + irrelevance}\n")
            all_irrelevance += irrelevance
            all_ground_truths.extend(ground_truths)
            all_predictions.extend(predictions)
    acc = accuracy_score(all_ground_truths, all_predictions)
    fout.write(f"=============================================\n")
    fout.write(f"[True-False Questions] Accuracy = {acc:.2f} ; "
               f"Irrelevant prediction = {all_irrelevance}/{len(all_ground_truths) + all_irrelevance}\n")
    fout.write(f"=============================================\n")

def eval_short_answer_question(data_dir, fout, template='single_answer'):
    all_total = 0
    all_correct = 0
    all_irrelevance = 0

    relations = ['date_of_birth', 'place_of_birth', 'place_of_death']
    for rel in relations:
        correct = 0
        total = 0
        irrelevance = 0
        pred_file = f"{args.data_dir}/output.{rel}.{template}.jsonl"
        with open(pred_file) as fin:
            for line in fin:
                data = json.loads(line.strip())
                ans = data['ground_truth'][0]['label']
                if rel == 'date_of_birth':
                    try:
                        predict = parse(data['predicted'].strip()).year
                        if predict == ans:
                            correct += 1
                        total += 1
                    except:
                        irrelevance += 1
                else:
                    predict = data['predicted'].strip()
                    if "I'm sorry" in predict or "cannot predict" in predict:
                        irrelevance += 1
                        continue
                    correct += int(utils.fuzzy_match(ans, predict))
                    total +=1
            if total > 0:
                acc = float(correct/total)
            else:
                acc = 0.
            fout.write(f"[{os.path.basename(pred_file)}] Accuracy = {acc:.2f} ; "
                       f"Irrelevant prediction = {irrelevance}/{total + irrelevance}\n")
            all_irrelevance += irrelevance
            all_total += total
            all_correct += correct
    acc = float(correct/total)
    fout.write(f"=============================================\n")
    fout.write(f"[True-False Questions] Accuracy = {acc:.2f} ; "
               f"Irrelevant prediction = {all_irrelevance}/{all_total + all_irrelevance}\n")
    fout.write(f"=============================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, help='Data dir')
    parser.add_argument('--out-file', type=str, help='Out file', default="result.txt")
    args = parser.parse_args()

    fout = open(f"{args.data_dir}/{args.out_file}", 'w')
    print(args)
    print("...Process true-false question results")
    eval_true_false_question(args.data_dir, fout)

    print("...Process short answer question results")
    eval_short_answer_question(args.data_dir, fout)
    fout.close()

