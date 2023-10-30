from certlm.language_models.base_language_model import BaseLanguageModel
from certlm.utils.util import LMResponse, LMInput
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

HUGGING_FACE_MODEL = ['gpt2']


class Vicuna(BaseLanguageModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--model', type=str, help="HUGGING FACE MODEL")
        parser.add_argument('--max-length', type=int, help="max length", default=200)
        parser.add_argument('--device', type=int, help="PLM device", default=-1)
        parser.add_argument('--hf-pipeline', type=str, help='Huggingface pipeline. Default is text-generation',
                            default='text-generation')

    def __init__(self, args):
        super().__init__(args)
        self.max_length = args.max_length
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModelForCausalLM.from_pretrained(args.model,
                                                          low_cpu_mem_usage=True, torch_dtype=torch.float16)
        if args.device >= 0:
            self.model.cuda()

    @torch.inference_mode()
    def generate_sentence(self, lm_input: LMInput) -> LMResponse:
        query = lm_input.prompt + '\n' + lm_input.message
        input_ids = self.tokenizer([query]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        output_ids = output_ids[0][len(input_ids[0]):]
        outputs = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return LMResponse(0, outputs)  # type: ignore