from certlm.lanugage_models.base_language_model import BaseLanguageModel
from certlm.utils.util import LMResponse, LMInput
from transformers import pipeline

HUGGING_FACE_MODEL = ['gpt2']

class HuggingFacePLM(BaseLanguageModel):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--model', type=str, help="HUGGING FACE MODEL", choices = HUGGING_FACE_MODEL)
        parser.add_argument('--max-length', type=int, help="max length", default=200)
        parser.add_argument('--device', type=int, help="PLM device", default=-1)

    def __init__(self, args):
        super().__init__(args)
        self.model = pipeline(model=args.model)
        self.max_length = args.max_length
        self.generator = pipeline('text-generation', model=args.model, device=args.device)
    def generate_sentence(self, lm_input: LMInput) -> LMResponse:
        query = lm_input.prompt + '\n' + lm_input.message
        outputs = self.generator(query, return_full_text=False, max_length = self.max_length)
        return LMResponse(0, outputs[0]['generated_text']) # type: ignore