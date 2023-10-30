from .chatgpt import ChatGPT
from .huggingface_plm import HuggingFacePLM

LANGUAGE_MODEL_REGISTRY = {
    'none': None,
    'chat_gpt': ChatGPT,
    'huggingface_plm': HuggingFacePLM
}

def get_language_model(name):
    return LANGUAGE_MODEL_REGISTRY[name]