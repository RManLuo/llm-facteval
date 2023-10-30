import json
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)
class BaseSampler(object):
    """
    Base KG dataset. Define how to load dataset and extract relations
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument('--input-file', required=True, type=str, help="Question input file")
        parser.add_argument('--output-file', required=True, type=str, help="Output file")
        parser.add_argument('--prompt-template', type=str, required=True, help="Path to prompt template file.")
        parser.add_argument('--prompt-context', type=str, choices=['none', 'relevance', 'irrelevance', 'antifactual'],
                            default='none', help="Type of context to add to prompt")

    def __init__(self, args):
        self.args = args
        self.prompt_template = self._read_prompt_template(args.prompt_template)

    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template

    def _predict(self, question):
        """
        Args: question record to the classifier
            question_record = {
                "question_id": question_id,
                "lm_input": {
                    "message": ...,
                    "prompt": ...,
                },
                "answers": answers,
                "question_type": question_type,
            }
        Return: dictionary of prediction result, ground_truth, predicted
        """
        raise NotImplementedError

    def _evaluate(self, groundtruths, predictions):
        """
            Compute evaluation metric such as accuracy
        """
        raise NotImplementedError

    def run_inference(self, input_file, output_file):
        """
            Run inference on input_file and save the prediction to output_file

        Args:
            input_file: test input file
            output_file: output file
        """
        ground_truths = []
        predictions = []

        cannot_parse = 0
        with open(input_file) as fin, open(output_file, 'w') as fout:
            for line in tqdm(fin):
                question = json.loads(line.strip())
                predicted_answer, ground_truth, predicted = self._predict(question)
                json_record = json.dumps(predicted_answer)
                if predicted >= 0:
                    ground_truths.append(ground_truth)
                    predictions.append(predicted)
                else:
                    cannot_parse += 1
                fout.write(json_record + '\n')
        logger.info("Finished prediction and start computing evaluation metrics")
        metric = self._evaluate(ground_truths, predictions)
        logger.info(f" * Eval metric: {metric}")
        logger.info(f" * Cannot parse: {cannot_parse}/{cannot_parse + len(ground_truths)}")
        return metric