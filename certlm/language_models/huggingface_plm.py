from certlm.language_models.base_language_model import BaseLanguageModel
from certlm.utils.util import LMResponse, LMInput
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

HUGGING_FACE_MODEL = ['gpt2']


class HuggingFacePLM(BaseLanguageModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--model', type=str, help="HUGGING FACE MODEL")
        parser.add_argument('--max-length', type=int, help="max length", default=200)
        parser.add_argument('--device', type=int, help="PLM device", default=-1)
        parser.add_argument('--hf-pipeline', type=str, help='Huggingface pipeline. Default is text-generation',
                            default='text-generation')
        parser.add_argument('--fp16', type=bool, action='store_true', default=False, help='Use fp16')

    def __init__(self, args):
        super().__init__(args)
        self.max_length = args.max_length
        if args.model.startswith('.') or args.model.startswith('/') or len(args.model.split('/')) > 2:
            if args.fp16:
                model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            self.generator = pipeline(args.hf_pipeline, model=model,
                                      tokenizer=tokenizer, device=args.device)
        else:
            self.generator = pipeline(args.hf_pipeline, model=args.model, device=args.device)

    @torch.inference_mode()
    def generate_sentence(self, lm_input: LMInput) -> LMResponse:
        query = lm_input.prompt + '\n' + lm_input.message
        # outputs = self.generator(query, return_full_text=False, max_length = self.max_length)
        if self.args.hf_pipeline == 'text-generation':
            outputs = self.generator(query, return_full_text=False, max_length=self.max_length)
        else:
            outputs = self.generator(query, max_length=self.max_length)
        return LMResponse(0, outputs[0]['generated_text'])  # type: ignore