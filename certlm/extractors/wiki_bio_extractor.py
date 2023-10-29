import json
import logging
import glob
from certlm.extractors.base_extractor import BaseExtractor
logger = logging.getLogger(__name__)


class WikiBioExtractor(BaseExtractor):
    def __init__(self, args):
        super().__init__(args)

    def load_data(self, filepath):
        data = []
        for file in glob.glob(filepath + "/*.jsonl"):
            with open(file, "r") as f:
                for line in f.readlines():
                    data.append(json.loads(line))
        return data

    def extract_relation(self, relation, relation_input, output):

        data = self.load_data(relation_input)
        logger.info(f"** Relation: {relation['relation']} **")
        logger.info(f"- Relation label: {relation['label']}")
        logger.info(f"- Relation type: {relation['type']}")
        logger.info(f"- Sample size: {len(data)}")
        relation["subject_symbol"] = "[X]"
        relation["object_symbol"] = "[Y]"
        object_nodes = {
        }
        subject_nodes = {
        }
        with open(output, 'w', encoding='utf-8') as fout:
            for sample in data:
                subject_label = sample["sub_label"]  # X
                subject_uri = sample["sub_uri"]
                for object_label, object_uri in zip(sample["obj_labels"], sample["obj_uris"]):  # Y
                    if object_uri not in object_nodes:
                        object_nodes[object_uri] = {
                            "object_label": object_label,
                            "object_uri": object_uri,
                            "subjects": []
                        }
                    if subject_uri not in subject_nodes:
                        subject_nodes[subject_uri] = {
                            "subject_label": subject_label,
                            "subject_uri": subject_uri,
                            "objects": []
                        }
                    object_nodes[object_uri]["subjects"].append({
                        "subject_label": subject_label,
                        "subject_uri": subject_uri,
                    })
                    subject_nodes[subject_uri]["objects"].append({
                        "object_label": object_label,
                        "object_uri": object_uri,
                    })

                    qa_record = {
                        'subject_label': subject_label,
                        'object_label': object_label,
                        "object_uri": object_uri,
                        "subject_uri": subject_uri,
                        "relation_info": relation
                    }
                    json_record = json.dumps(qa_record, ensure_ascii=False)
                    fout.write(json_record + '\n')

        node_output = '.'.join(output.split('.')[:-1])
        node_output = f"{node_output}.summary.json"
        relation_summary = {
            "relation_info": relation,
            "node_summary": {
                "objects": object_nodes,
                "subjects": subject_nodes
            }
        }
        with open(node_output, 'w', encoding='utf-8') as fout:
            json.dump(relation_summary, fout)

    def get_input_question_files(self, data_dir, relation):
        question_input = f"{data_dir}/{relation['relation']}.jsonl"
        summary = f"{data_dir}/{relation['relation']}.summary.json"
        return question_input, summary