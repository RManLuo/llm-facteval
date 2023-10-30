from certlm.answer_samplers.multi_choice_qa_sampler import MCQAnswerSampler
from certlm.answer_samplers.short_qa_sampler import ShortQAnswerSampler
from certlm.answer_samplers.true_false_qa_sampler import TrueFalseQAnswerSampler

ANS_SAMPLER_REGISTRY = {
 'true_false': TrueFalseQAnswerSampler,
 'mcq-qa': MCQAnswerSampler,
 'short_answer': ShortQAnswerSampler,
}


def get_answer_sampler(name):
    return ANS_SAMPLER_REGISTRY[name]