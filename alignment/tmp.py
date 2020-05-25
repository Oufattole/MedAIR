from ..tokenizers import CoreNLPTokenizer
from ..retriever import TfidfDocRanker
from ..retriever import DocDB
from .. import DATA_DIR

DEFAULTS = {
    'tokenizer': CoreNLPTokenizer, 
    'ranker': TfidfDocRanker,
    'db': DocDB, # database for docs (..)
    'reader_model': os.path.join(DATA_DIR, 'reader/multitask.mdl'),
}