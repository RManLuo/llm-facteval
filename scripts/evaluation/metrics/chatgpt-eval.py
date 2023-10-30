import argparse
import json

import numpy as np
from tqdm import tqdm
import os
from certlm.language_models import ChatGPT
from certlm.language_models.chatgpt import OPENAI_MODEL
from certlm.utils.util import LMInput

def test_unknown_response(unk_list, response):
    for unk in unk_list:
        if unk in response:
            return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=str, help='Reference file', default='ref.txt')
    parser.add_argument('--hypothesis', type=str, help='Hypothesis file', default="hyp.txt")
    parser.add_argument('--out-file', type=str, help='Output file', default="out.txt")
    parser.add_argument('--open-ai-key', type=str, help="openai_api_key")
    parser.add_argument('--model', type=str, help="Chat GPT model", choices=OPENAI_MODEL)
    parser.add_argument('--retry', type=int, help="retry time", default=3)
    args = parser.parse_args()

    unk_file = f"{os.getcwd()}/unknown_response.txt"
    print(f"Load unknown response template from {unk_file}")
    unknown_responses = []
    with open(unk_file) as fin:
        for line in fin:
            unknown_responses.append(line.strip())

    choices = ['Yes', 'No']
    prompt = f"Answer the question by printing only a single choice from {choices} (without quotes or punctuation) corresponding to the correct answer with no other text."
    prompt = prompt.format(','.join(choices))
    chatgpt = ChatGPT(args)
    answer_map = {
        'Yes': 1,
        'No': 0
    }
    response = []
    with open(args.reference) as fref, open(args.hypothesis) as fhyp, open(args.out_file, 'w') as fout:
        for ref, hyp in tqdm(zip(fref, fhyp)):
            ref_data = json.loads(ref.strip())
            hyp_data = json.loads(hyp.strip())
            triplet_id = ':'.join(hyp_data['question_id'].split(':')[:3])
            assert ref_data['triplet_id'] == hyp_data['question_id']
            hyp_sent = hyp_data['predicted'].replace('/n', ' ')
            ref_sent = ref_data['reference'].replace('/n', ' ')
            if test_unknown_response(unknown_responses, hyp_sent):
                fout.write(f"Unk\n")
                continue
            message = f"""Sentence 1: {ref_sent}
            Sentence 2: {hyp_sent}
            """
            lm_input = LMInput(message, prompt)
            res = chatgpt.generate_sentence(lm_input)
            msg = res.message.strip()
            fout.write(f"{msg}\n")
            response.append(answer_map[msg])
    response = np.asarray(response)
    acc = np.mean(response)
    print(f"Accuracy {acc:.2f}")
