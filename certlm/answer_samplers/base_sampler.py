import logging

logger = logging.getLogger(__name__)
class BaseSampler(object):
    """
    Base KG dataset. Define how to load dataset and extract relations
    """
    @staticmethod
    def add_args(parser):
        pass

    def __init__(self, args):
        self.args = args

    def load_data(self):
        raise NotImplementedError