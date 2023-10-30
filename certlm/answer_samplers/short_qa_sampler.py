import logging

from sklearn.metrics import accuracy_score

from certlm.answer_samplers.base_sampler import BaseSampler
from certlm.language_models import get_language_model, CHATGPT
from certlm.utils.util import LMInput

logger = logging.getLogger(__name__)
class ShortQAnswerSampler(BaseSampler):
    @staticmethod
    def add_args(parser):
        BaseSampler.add_args(parser)

    def __init__(self, args):
        super().__init__(args)
        self.generator = get_language_model(self.args.eval_lm)(args)

    def _predict(self, question_dict):
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
        Return: dictionary of prediction result
        """
        question = question_dict['lm_input']['message']
        answer = question_dict['answers']
        context = ""
        if self.args.prompt_context != 'none' and 'contexts' in question_dict:
            context = question_dict['contexts'][self.args.prompt_context]['snippet']
        prompt = self.prompt_template.format(**locals())
        if self.args.eval_lm == CHATGPT:
            lm_input = LMInput(question_dict['lm_input']['message'], prompt)
        else:
            lm_input = LMInput(prompt, '')
        res = self.generator.generate_sentence(lm_input)
        predicted = res.message
        output = {
            'ground_truth': answer,
            'predicted': predicted,
            'status_code': res.status_code,
            'question_id': question_dict['question_id'],
            'lm_input': {
                'prompt': lm_input.prompt,
                'message': lm_input.message
            }
        }
        return output, 0, 0


    def _evaluate(self, groundtruths, predictions):
        """
            Compute evaluation metric such as accuracy
        """
        acc = accuracy_score(groundtruths, predictions)
        return acc