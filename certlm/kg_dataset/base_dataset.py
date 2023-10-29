import logging
from enum import Enum

logger = logging.getLogger(__name__)

class BaseKG(object):
    """
    Base KG dataset. Define how to load dataset and extract relations
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument('--data-dir', type=str, help="Data directory")
        parser.add_argument('--rel-file', type=str, help="Relation data file")
        parser.add_argument('--registry', type=str, help="Path to registry file")

    def __init__(self, args):
        self.args = args
        self.relations = []
    
    def load_relations(self):
        """
        Load list relations
        """
        raise NotImplementedError

    def get_input_relation_file(self, relation):
        raise NotImplementedError

    def get_relation_type(self, relation):
        raise NotImplementedError

    def get_relation_name(self, relation):
        raise NotImplementedError