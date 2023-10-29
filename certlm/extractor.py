import collections
import logging
import multiprocessing
import os
import pickle
import glob
import time
import json

import certlm.registry as registry
logger = logging.getLogger(__name__)

ExtractorArgs = collections.namedtuple('ExtractorArgs', ['relation_index', 'relation', 'relation_input',
                                                         'kg_name', 'output_file'])

def extract_tuples(extract_args):
    pid = os.getpid()
    def log(*args):
        msg = " ".join(map(str, args))
        print(f"Process {pid}: {msg}", flush=True)
    relation_input = extract_args.relation_input
    relation = pickle.loads(extract_args.relation)
    output_file = extract_args.output_file
    log(f"Start extracting {relation_input}")
    start = time.time()
    kg_extractor = registry.get_extractor(extract_args.kg_name)(None)
    kg_extractor.extract_relation(relation, relation_input, output_file)
    log(f"Finish extracting {relation_input} in {(time.time() - start) / 60:.2f} "
        f"minutes")


def run_extract(kg, num_workers=1):
    logger.info(f"Start extracting relation tuples using {num_workers} worker(s)")
    worker_inputs = []
    args = kg.args
    output_dir = f"{args.output_dir}/tuples"
    os.makedirs(output_dir, exist_ok=True)
    for idx, relation in enumerate(kg.relations):
        output_file = f"{output_dir}/{kg.get_relation_name(relation)}.jsonl"
        if not os.path.exists(kg.get_input_relation_file(relation)):
            logger.info(f"{kg.get_input_relation_file(relation)} does not exist. Skip")
            continue
        worker_inputs.append(ExtractorArgs(idx, pickle.dumps(relation), kg.get_input_relation_file(relation),
                                           args.kg, output_file))

    start = time.time()
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        pool.map(extract_tuples, worker_inputs)
    end = time.time()
    logger.info(f"Finish extract all relations in {(end - start) / 60:.2f} minutes")
    
    # Summary results
    entities = {}
    n_facts = 0
    for relation_file in glob.glob(f"{output_dir}/*.jsonl"):
        with open(relation_file, "r") as f:
            for line in f.readlines():
                n_facts += 1
                data = json.loads(line)
                subject_label = data["subject_label"]
                object_label = data["object_label"]
                subject_uri = data["subject_uri"]
                object_uri = data["object_uri"]
                if subject_uri not in entities:
                    entities[subject_uri] = subject_label
                if object_uri not in entities:
                    entities[object_uri] = object_label
    logger.info(f"Total number of entities: {len(entities)}")
    logger.info(f"Total number of relations: {len(kg.relations)}")
    with open(f"{args.output_dir}/summary.json", "w") as f:
        summary = {'num_entities': len(entities), 'num_relations': len(kg.relations), 'n_facts': n_facts, 'entities': entities, 'relations': kg.relations}
        json.dump(summary, f, indent=4) 