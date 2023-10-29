import collections

LMResponse = collections.namedtuple('LMResponse', ['status_code', 'message'])

LMInput = collections.namedtuple('LMInput', ['message', 'prompt'])
