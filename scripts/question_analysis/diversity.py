import evaluate
import argparse
import os
import glob
import json
import copy
from selfBlue import SelfBleu

bleu = evaluate.load("bleu")

def compute_bleu(sentence, questions):
    questions_copy = copy.deepcopy(questions)
    questions_copy.remove(sentence)
    s = bleu.compute(predictions=sentence, references=questions_copy)
    return s['bleu']

def main(args):

    overall_results = {"tfq": [], "cloze": [], "SAQ": []}

    rel_list = glob.glob(os.path.join(args.data, "*.jsonl"))
    for rel in rel_list:
        questions = {"tfq": [], "cloze": [], "SAQ": []}
        with open(rel, "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                if data['question_type'] != "multi_choices":
                    if "qa" in args.data:
                        questions["SAQ"].append(data['lm_input']['message'])
                    else:
                        if data['question_type'] == "single_answer" or data['question_type'] == "multiple_answer":
                            questions["cloze"].append(data['lm_input']['message'])
                        elif "binary" in data['question_type']:
                            questions["tfq"].append(data['lm_input']['message'])
                    
        # Self-bleu

        for key in questions:
            if len(questions[key]) == 0:
                continue
            self_blue = SelfBleu(questions[key])
            if args.p:
                score = self_blue.get_bleu_parallel()
            else:
                score = self_blue.get_bleu()
            #print(f"rel: {rel}, type: {key}, Self-Bleu: {score}")
            
            overall_results[key].append(score)
            
    for key in overall_results:
        if len(overall_results[key]) == 0:
            continue
        print(f"Overall {args.data} {key}: ", sum(overall_results[key]) / len(overall_results[key]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="Data directory")
    parser.add_argument('-p', action='store_true', help="Parallel")

    args = parser.parse_args()
    main(args)