
from .trex import TrexKG
from .google_re import GoogleREKG
from .umls import UmlsKG
from .ctd import CTDKG
from .wiki_bio import WikiBioKG

KG_REGISTRY = {
    'trex': TrexKG,
    'google_re': GoogleREKG,
    'umls': UmlsKG,
    'ctd': CTDKG,
    'wiki_bio': WikiBioKG,
}

def get_kg(name):
    return KG_REGISTRY[name]