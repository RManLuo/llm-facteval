import argparse
import glob
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='Path to question_template')
    parser.add_argument('--out-dir', type=str, help='Out dir')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for template in ['binary_answer', 'single_answer', 'multiple_answer']:
        all_questions = sorted(glob.glob(f"{args.data_dir}/*.{template}.jsonl"))
        for question_file in all_questions:
            print(f"Process {question_file}")
            basename = '.'.join(os.path.splitext(os.path.basename(question_file))[:-1])
            with open(f"{args.out_dir}/{basename}.jsonl", 'w') as fout, open(question_file) as fin:
                for line in fin:
                    question = json.loads(line)
                    answer = question['answers']
                    if answer == 'incorrect':
                        continue
                    question_id = question['question_id'].split(':')
                    gt_id = ':'.join(question_id[:3])
                    data = {
                        'triplet_id': gt_id,
                        'reference': question['lm_input']['message']
                    }
                    json_record = json.dumps(data)
                    fout.write(json_record + '\n')

