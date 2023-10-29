import logging
import os
import sys
import certlm.registry as registry
from certlm import options
from certlm.extractor import run_extract
from certlm.question_generator import run_generation

logging.basicConfig(
  format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
  level=os.environ.get("LOGLEVEL", "INFO").upper(),
  stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def extract_tuples(args):
  kg = registry.get_kg(args.kg)(args)
  kg.load_relations()
  logger.info(f"Loaded {len(kg.relations)} relations in knowledge graph {args.kg}")
  run_extract(kg, args.num_workers)

def generate_question(args):
  generator = registry.get_question_generator(args.generator)(args)
  kg = registry.get_kg(args.kg)(args)
  extractor = registry.get_extractor(args.kg)(args)
  kg.load_relations()
  logger.info(f"Loaded {len(kg.relations)} relations in knowledge graph {args.kg}")
  run_generation(generator, extractor, kg, args.num_workers)

def sampling_answer(args):
  pass

def main_cli():
  args = options.get_args()
  logger.info(args)
  if args.step == 'extract':
    extract_tuples(args)
  elif args.step == 'question_generate':
    generate_question(args)
  elif args.step == 'answer_sampling':
    sampling_answer(args)

if __name__ == '__main__':
    main_cli()