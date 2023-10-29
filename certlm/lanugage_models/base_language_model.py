
from certlm.utils.util import LMResponse, LMInput
from typing import Union
class BaseLanguageModel(object):
    """
    Base lanuage model. Define how to generate sentence by using a LM
    Args:
        args: arguments for LM configuration
    """
    @staticmethod
    def add_args(parser):
        return
    
    def __init__(self, args):
        self.args = args
        
    def generate_sentence(self, lm_input: LMInput) -> LMResponse:
        '''
        Generate sentence by using a LM

        Args:
            lm_input (LMInput): input for LM

        Raises:
            NotImplementedError: _description_

        Returns:
            LMResponse: _description_
        '''
        raise NotImplementedError