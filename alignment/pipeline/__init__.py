import os
from ..tokenizers import SpacyTokenizer
from ..retriever import TfidfDocRanker
from ..retriever import DocDB
from .. import DATA_DIR

DEFAULTS = {
    'tokenizer': SpacyTokenizer,
    'ranker': TfidfDocRanker,
    'db': DocDB,
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value