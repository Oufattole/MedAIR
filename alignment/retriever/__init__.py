import os
from .. import DATA_DIR

DEFAULTS = {
    'db_path': DATA_DIR + '/corpus.db',
    'tfidf_path': DATA_DIR + '/corpus-tfidf-ngram=1-hash=16777216-tokenizer=corenlp.npz'
}

def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'sqlite':
        return DocDB
    raise RuntimeError('Invalid retriever class: %s' % name)

from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker