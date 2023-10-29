import collections
import logging
import multiprocessing
import os
import pickle

import certlm.registry as registry
from functools import partial
logger = logging.getLogger(__name__)

QuesGenArgs = collections.namedtuple('QuesGenArgs', ['question_id', 'relation', 'question_input', 'relation_summary',
                                                     'generator_name', 'kg_name', 'output_file'])


def generate_questions(gen_args, generator):
    pid = os.getpid()
    def log(*args):
        msg = " ".join(map(str, args))
        print(f"Process {pid}: {msg}", flush=True)
    question_input = gen_args.question_input
    relation_summary = gen_args.relation_summary
    relation = pickle.loads(gen_args.relation)
    output_file = gen_args.output_file
    log(f"Start generate question from {question_input}")
    generator.generate_questions(question_input, relation_summary, output_file)

def run_generation(generator, extractor, kg, num_workers=1):
    logger.info(f"Start extracting relation tuples using {num_workers} worker(s)")
    worker_inputs = []
    args = kg.args
    output_dir = f"{args.output_dir}/questions_{args.generator}"
    input_dir = f"{args.gen_input_dir}"
    os.makedirs(output_dir, exist_ok=True)
    for idx, relation in enumerate(kg.relations):
        output_file = f"{output_dir}/{kg.get_relation_name(relation)}.jsonl"
        tuple_file, summary_file = extractor.get_input_question_files(input_dir, relation)
        if not os.path.exists(tuple_file):
            logger.info(f"{tuple_file} does not exist. Skip")
            continue
        worker_inputs.append(QuesGenArgs(idx, pickle.dumps(relation), tuple_file, summary_file, args.generator,
                                           args.kg, output_file))
    # Test
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        pool.map(partial(generate_questions, generator = generator), worker_inputs)