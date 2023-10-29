from .qa_generator import QAGenerator
from .masking_generator import MaskingGenerator
from .template_generator import TemplateGenerator

QUES_GEN_REGISTRY = {
    'template': TemplateGenerator,
    'qa': QAGenerator,
    'masking': MaskingGenerator,
}

def get_question_generator(name):
    return QUES_GEN_REGISTRY[name]