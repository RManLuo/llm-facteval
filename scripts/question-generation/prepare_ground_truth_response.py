import argparse
import glob
import json
import os

def load_reference_file(ref_file):
    data = {}
    with open(ref_file) as fin:
        for line in fin:
            ref = json.loads(line.strip())
            qid = ref['triplet_id']
            sent = ref['reference']
            data[qid] = line
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='data dir')
    parser.add_argument('--question-dir', type=str, help='Out dir')
    parser.add_argument('--out-dir', type=str, help='Out dir')
    parser.add_argument('--reference-template', type=str, help='reference template',
                        default="binary_answer")
    args = parser.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    all_questions = sorted(glob.glob(f"{args.question_dir}/*.jsonl"))
    for question_file in all_questions:
        print(f"Process {question_file}")
        basename = '.'.join(os.path.splitext(os.path.basename(question_file))[:-1])
        relation_name = '.'.join(basename.split('.')[:-1])
        reference_file = f"{args.data_dir}/{relation_name}.{args.reference_template}.jsonl"
        data = load_reference_file(reference_file)
        with open(f"{out_dir}/{basename}.jsonl", 'w') as fout, open(question_file) as fin:
            for line in fin:
                question = json.loads(line)
                question_id = question['question_id'].split(':')
                gt_id = ':'.join(question_id[:3])
                if gt_id in data:
                    fout.write(data[gt_id])
                else:
                    d = {
                        'triplet_id': gt_id,
                        'reference': ''
                    }
                    fout.write(f"{json.dumps(d)}\n")