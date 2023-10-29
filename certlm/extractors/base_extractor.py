
class BaseExtractor:
    def __init__(self, args):
        self.args = args

    def load_data(self, file):
        raise NotImplementedError

    def extract_relation(self, relation, relation_input, output):
        """
        Extract triplets for a relation
        relation: relation info
        relation_input: path to triplet data
        output: output file
        """
        raise NotImplementedError

    def get_input_question_files(self, data_dir, relation):
        """
        Return tuple inputs and relation summary
        """
        raise NotImplementedError