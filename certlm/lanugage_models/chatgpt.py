import time

import openai

from certlm.lanugage_models.base_language_model import BaseLanguageModel
from certlm.utils.util import LMResponse, LMInput

OPENAI_MODEL = ['gpt-4', 'gpt-3.5-turbo']

class ChatGPT(BaseLanguageModel):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--open-ai-key', type=str, help="Open AI key")
        parser.add_argument('--model', type=str, help="Chat GPT model", choices = OPENAI_MODEL)
        parser.add_argument('--retry', type=int, help="retry time", default=3)
    
    def __init__(self, args):
        super().__init__(args)
        self.api_key = args.open_ai_key
        self.model = args.model
        self.retry = args.retry

    def _create_input(self, lm_input: LMInput) -> list:
        query = []
        if lm_input.prompt is not None:
            query.append({"role": "system",
                       "content": lm_input.prompt})
        query.append({"role": "user", "content": lm_input.message})
        return query

    def generate_sentence(self, lm_input: LMInput) -> LMResponse:
        query = self._create_input(lm_input)
        cur_retry = 0
        num_retry = self.retry
        while cur_retry <= num_retry:
            try:
                response = openai.ChatCompletion.create(
                    api_key = self.api_key,
                    model=self.model,
                    messages= query,
                    request_timeout = 30,
                    )
                result = response["choices"][0]["message"]["content"].strip() # type: ignore
                return LMResponse(0, result)
            except Exception as e:
                print(e)
                time.sleep(10)
                cur_retry += 1
                continue
        return LMResponse(1, f"Failed after {num_retry} retries")