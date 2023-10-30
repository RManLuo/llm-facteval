
import logging
logger = logging.getLogger(__name__)


def run_answer_sampling(sampler):
    args = sampler.args
    metrics = sampler.run_inference(args.input_file, args.output_file)
