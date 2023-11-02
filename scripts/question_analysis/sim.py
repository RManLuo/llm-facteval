from evaluate import load
import argparse
import os
import glob
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help="Data directory")

args = parser.parse_args()

bertscore = load("bertscore")

template_path = os.path.join(args.data, "questions_template")
predictions_path = os.path.join(args.data, "questions_masking")
rel_list = glob.glob(os.path.join(template_path, "*.jsonl"))

overall_results = {"tfq": [], "cloze": []}

for rel in rel_list:
    rel_name = os.path.basename(rel).split(".")[0]
    template = {}
    with open(rel, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            template[data['question_id']] = data
    predictions = {}
    with open(os.path.join(predictions_path, f"{rel_name}.jsonl"), "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            predictions[data['question_id']] = data
    references = {"tfq": [], "cloze": []}
    pred = {"tfq": [], "cloze": []}
    for qid in predictions:
        if qid in template:
            if template[qid]['question_type'] == "single_answer" or template[qid]['question_type'] == "multiple_answer":
                references["cloze"].append(template[qid]['lm_input']['message'])
                pred["cloze"].append(predictions[qid]['lm_input']['message'])
            elif "binary" in template[qid]['question_type']:
                references["tfq"].append(template[qid]['lm_input']['message'])
                pred["tfq"].append(predictions[qid]['lm_input']['message'])

    for key in references:
        if len(references[key]) == 0:
            continue
        
        results = bertscore.compute(predictions=pred[key], references=references[key], lang="en", device="mps", nthreads=8)
    
        score = sum(results['f1'])/len(results['f1'])
        #print(f"rel: {rel}, type: {key}, Sim: {score}")
            
        overall_results[key].append(score)

for key in overall_results:
        if len(overall_results[key]) == 0:
            continue
        print(f"Overall {args.data} {key}: ", sum(overall_results[key]) / len(overall_results[key]))