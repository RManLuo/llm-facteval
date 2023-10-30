import argparse
import pandas as pd
MODELS = ['chatgpt', 'llama-7b', 'alpaca-7b-fp16', 'vicuna-7b-fp16',
           't5-xl', 'flan-t5', 'flan-t5-xl', 'flan-alpaca-xl', 'fastchat-t5-3b']

EXP_MAP = {
  'tfq-template': [('template','binary_answer')],
  'tfq-masking': [('masking', 'binary_answer')],
  'cloze-template': [('template','single_answer'), ('template','multiple_answer')],
  'cloze-masking': [('masking','single_answer'), ('masking','multiple_answer')],
  'mcq-template': [('template', 'multi_choices')],
  'mcq-masking': [('masking', 'multi_choices')],
  'mcq-qa': [('qa', 'multi_choices')],
  'asq-qa': [('qa','single_answer'), ('qa','multiple_answer')]
}

EXPS = ['tfq-template', 'tfq-masking',
        'mcq-template', 'mcq-masking', 'mcq-qa', 'cloze-template', 'cloze-masking', 'asq-qa']


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


def read_result_file(result_file):
  try:
    data = pd.read_csv(result_file)
    all = data[data.relation == 'all']
    precision = all['precision'].values[0]
    recall = all['recall'].values[0]
    f1 = all['f1'].values[0]
    num_invalid = all['invalid'].values[0]
    num_abstain = all['abstain'].values[0]
    num_correct = all['correct'].values[0]
    num_incorrect = all['incorrect'].values[0]
    total = all['size'].values[0]
    return precision, recall, f1, num_correct, num_incorrect, num_abstain, num_invalid, total
  except:
    print(f"Error in parsing {result_file}")
    return 0, 0, 0, 0, 0, 0, 0, 0


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', type=str, help='Path to experiment results')
  parser.add_argument('--out-file', type=str, help='Out file', default="test")
  args = parser.parse_args()
  precisions = {}
  recalls = {}
  f1s = {}
  invalid_ratios = {}
  abstained_ratios = {}
  sizes = {}
  prompt_context = 'none'
  for model in MODELS:
    p = {}
    r = {}
    f = {}
    iratio = {}
    aratio = {}
    s = {}
    for exp_name in EXPS:
      total_correct, total_incorrect, total_abstain, total_invalid, total_size = 0, 0, 0, 0, 0
      for config in EXP_MAP[exp_name]:
        result_file = f"{args.data_dir}/questions_{config[0]}/{model}-{prompt_context}/result.{config[1]}.txt.detail.csv"
        precision, recall, f1, num_correct, num_incorrect, num_abstain, num_invalid, total = read_result_file(result_file)
        total_correct += num_correct
        total_incorrect += num_incorrect
        total_abstain += num_abstain
        total_invalid += num_invalid
        total_size += total
      s[exp_name] = total_size
      if total_size == 0:
        p[exp_name] = -1
        r[exp_name] = -1
        f[exp_name] = -1
        iratio[exp_name] = -1
        aratio[exp_name] = -1
      else:
        precision, recall, f1 = calculate_f1(total_correct, total_incorrect, total_abstain)
        p[exp_name] = precision
        r[exp_name] = recall
        f[exp_name] = f1
        iratio[exp_name] = float(total_invalid / total_size) * 100
        aratio[exp_name] = float(total_abstain / total_size) * 100
    precisions[model] = p
    recalls[model] = r
    f1s[model] = f
    invalid_ratios[model] = iratio
    abstained_ratios[model] = aratio
    sizes[model] = s


  with open(f"{args.out_file}.precision.csv", 'w') as fprecision, \
      open(f"{args.out_file}.recall.csv", 'w') as frecall, \
      open(f"{args.out_file}.f1.csv", 'w') as ff1, \
      open(f"{args.out_file}.iratio.csv", 'w') as fratio, \
      open(f"{args.out_file}.aratio.csv", 'w') as fabs, \
      open(f"{args.out_file}.size.csv", 'w') as fsize:
    fprecision.write(f"model,{','.join(EXPS)}\n")
    frecall.write(f"model,{','.join(EXPS)}\n")
    ff1.write(f"model,{','.join(EXPS)}\n")
    fratio.write(f"model,{','.join(EXPS)}\n")
    fabs.write(f"model,{','.join(EXPS)}\n")
    fsize.write(f"model,{','.join(EXPS)}\n")
    for model in MODELS:
      fprecision.write(f"{model},")
      frecall.write(f"{model},")
      ff1.write(f"{model},")
      fratio.write(f"{model},")
      fabs.write(f"{model},")
      fsize.write(f"{model},")
      for exp_name in EXPS:
        fprecision.write(f"{precisions[model][exp_name]:.2f},")
        frecall.write(f"{recalls[model][exp_name]:.2f},")
        ff1.write(f"{f1s[model][exp_name]:.2f},")
        fratio.write(f"{invalid_ratios[model][exp_name]:.2f},")
        fabs.write(f"{abstained_ratios[model][exp_name]:.2f},")
        fsize.write(f"{sizes[model][exp_name]:.2f},")
      fprecision.write('\n')
      frecall.write('\n')
      ff1.write('\n')
      fratio.write('\n')
      fabs.write('\n')
      fsize.write('\n')