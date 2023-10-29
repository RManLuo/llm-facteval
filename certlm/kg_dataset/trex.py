import json

from certlm.kg_dataset.base_dataset import BaseKG


class TrexKG(BaseKG):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--data-dir', type=str, help="Data directory")
        parser.add_argument('--rel-file', type=str, help="Relation file", default="relations.jsonl")
        parser.add_argument('--registry', type=str, help="Path to registry file")
    def load_relations(self):
        self.relations = self._load_file(f"{self.args.data_dir}/{self.args.rel_file}")

    def _load_file(self, filename):
        data = []
        with open(filename, "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))
        return data

    def get_input_relation_file(self, relation):
        relation_name = relation["relation"]
        return f"{self.args.data_dir}/{relation_name}.jsonl"

    def get_relation_type(self, relation):
        return relation['type']

    def get_relation_name(self, relation):
        return relation['relation']
