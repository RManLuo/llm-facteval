import collections
import json
import logging

logger = logging.getLogger(__name__)

Question = collections.namedtuple('Question', ['question_id', 'prompts', 'answers'])
class BaseGenerator(object):
    """
    Base KG dataset. Define how to load dataset and extract relations
    """
    MASK_TOKEN = "[MASK]"
    @staticmethod
    def add_args(parser):
        parser.add_argument('--gen-input-dir', type=str, help="Data directory")
        parser.add_argument('--thread', type=int, help="Number of threads for generating questions", default=1)
        parser.add_argument('--force', action='store_true', help="Force to overwrite existing files")
        parser.add_argument('--debug', action='store_true', help="debug")

    def __init__(self, args):
        self.args = args
        self.thread = args.thread
        self.force = args.force
        self.generated = []
        self.debug = args.debug
        
    def generate_questions(self, tuple_input, relation_summary, output_file):
        raise NotImplementedError

    def _generate_question_id(self, subject_uri, object_uri, relation_id, relation_type, suffix):
        parts = [subject_uri, object_uri, relation_id, relation_type, suffix]
        return ':'.join(parts)

    def _load_relation_summary(self, relation_summary):
        data = None
        with open(relation_summary, 'r') as fin:
            data = json.load(fin)
        return data

    def write_question(self, fout, question_id, prompts, answers, question_type):
        if question_id in self.generated:
            if self.debug:
                logger.info(f"Question {question_id} already generated")
            return
        question_record = {
            "question_id": question_id,
            "lm_input": prompts,
            "answers": answers,
            "question_type": question_type,
        }
        json_record = json.dumps(question_record)
        fout.write(json_record + '\n')