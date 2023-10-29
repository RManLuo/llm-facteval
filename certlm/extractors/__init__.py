from .trex_extractor import TrexExtractor
from .google_re_extractor import GoogleREExtractor
from .umls_extractor import UmlsExtractor
from .ctd_extractor import CTDExtractor
from .wiki_bio_extractor import WikiBioExtractor

EXTRACTOR_REGISTRY = {
    'trex': TrexExtractor,
    'google_re': GoogleREExtractor,
    'umls': UmlsExtractor,
    'ctd': CTDExtractor,
    'wiki_bio': WikiBioExtractor,
}


def get_extractor(name):
    return EXTRACTOR_REGISTRY[name]