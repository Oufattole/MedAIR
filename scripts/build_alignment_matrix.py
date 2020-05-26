#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to build the tf-idf document matrices for retrieval."""

import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from tqdm import tqdm
from functools import partial
from collections import Counter
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors

from alignment import retriever
from alignment import tokenizers
from alignment import DATA_DIR
from alignment.retriever import utils
from alignment.retriever.question import Question

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)

def fetch_filename(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_filename(doc_id)


def tokenize(text):
    global PROCESS_TOK
    assert(not PROCESS_TOK is None)
    return PROCESS_TOK.tokenize(text)

def _tk(doc_id):
    text = PROCESS_DB.get_doc_text(doc_id)
    return set(tokenize(utils.normalize(text)).words())


# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------

def load_emb(name):
    if name == "glove":
        return load_glove_emb()
    elif name == "bio-wv":
        return load_biowv()
    elif name == "wiki-pubmed-pmc-wv":
        return load_bio_wiki()
    elif name == "pubmed-pmc-wv":
        return load_bio()
    else:
        raise RuntimeError("Embedding type not found")

def load_glove_emb():
    df = pd.read_csv(DATA_DIR+'/embeddings/glove.840B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
    glove = {key: val.values for key, val in df.T.items()}
    return glove

def load_biowv():
    wv_from_bin = KeyedVectors.load_word2vec_format(DATA_DIR+"/embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin", binary = True)  # C bin format
    return wv_from_bin

def load_bio_wiki():
    wv_from_bin = KeyedVectors.load_word2vec_format("/embeddings/wikipedia-pubmed-and-PMC-w2v.bin", binary=True)  # C bin format
    return wv_from_bin

def load_bio():
    wv_from_bin = KeyedVectors.load_word2vec_format("/embeddings/PubMed-and-PMC-w2v.bin", binary=True)  # C bin format
    return wv_from_bin

def get_question_tokens():
    questions_filepath = DATA_DIR + "/questions"
    question_files = os.listdir(questions_filepath)
    question_tokens = set()
    for filename in question_files:
        # count = 0
        questions = Question.read_jsonl(questions_filepath+"/"+filename)
        for question in questions:
            question_tokens.update(set(tokenize(utils.normalize(question.get_prompt())).words()))
            for option in question.get_options():
                question_tokens.update(set(tokenize(utils.normalize(option)).words()))
            # print(f"{filename}: {count/len(questions)}")
            # count+=1
    return question_tokens

def get_corpus_tokens(args):
    workers = ProcessPool(
        args.num_workers)
    tokenizer = PROCESS_TOK
    doc_db = PROCESS_DB
    doc_ids = doc_db.get_doc_ids()
    paragraph_tokens = set()
    for token_set in workers.imap_unordered(_tk, doc_ids):
            paragraph_tokens.update(token_set)
    workers.close()
    workers.join()
    # for doc_id in doc_ids:
    #     text = doc_db.get_doc_text(doc_id)
    #     paragraph_tokens.update(set(tokenize(utils.normalize(text)).words()))
    return paragraph_tokens
# q tokens: 36651
# p tokens: 281835


def count(ngram, hash_size, doc_id):
    """Fetch the text of a document and compute hashed ngrams counts."""
    global DOC2IDX
    row, col, data = [], [], []
    # Tokenize    
    tokens = tokenize(retriever.utils.normalize(fetch_text(doc_id)))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(
        n=ngram, uncased=True, filter_fn=retriever.utils.filter_ngram
    )

    # Hash ngrams and count occurences
    counts = Counter([retriever.utils.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend(counts.keys())
    col.extend([DOC2IDX[doc_id]] * len(counts))
    data.extend(counts.values())
    return row, col, data


def get_count_matrix(args, db, db_opts):
    """Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    # Map doc_ids to indexes
    global DOC2IDX
    db_class = retriever.get_class(db)
    with db_class(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()
    DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    # Setup worker pool
    tok_class = tokenizers.get_class(args.tokenizer)
    workers = ProcessPool(
        args.num_workers
    )

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, args.ngram, args.hash_size)
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
    workers.close()
    workers.join()

    logger.info('Creating sparse matrix...')
    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_ids))
    )
    count_matrix.sum_duplicates()
    return count_matrix, (DOC2IDX, doc_ids)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------
def get_doc_matrix(embedding, doc_id):
    text = PROCESS_DB.get_doc_text(doc_id)
    tokens = PROCESS_TOK.tokenize(text).words()
    matrix = None
    # logging.info(f"doc text: {text}")
    for token in tokens:
        try:
            token_emb = np.array([embedding[token]])
        except:
            # logging.info(f"failed to encode: {token}")
            continue
        token_emb = token_emb/np.linalg.norm(token_emb)
        if matrix is None:
            matrix = token_emb
        else:
            matrix = np.vstack((matrix, token_emb))
    return matrix
def score(encoder, hash_size, batch):
    doc_ids = PROCESS_DB.get_doc_ids()
    row, col, data = [], [], []
    for doc_id in doc_ids:
        b_row, b_col, b_data = score_doc(encoder,hash_size, batch, doc_id)
        row.extend(b_row)
        col.extend(b_col)
        data.extend(b_data)
    return row, col, data

def generate_question_token_matrix(encoder, hash_size, question_tokens):
    q_mat = []
    q_hashes = []
    for token in question_tokens:
        try:
            embedded_token = np.array(encoder[token])
        except:
            continue
        q_hashes.append(retriever.utils.hash(token, hash_size))
        embedded_token = embedded_token/np.linalg.norm(embedded_token)
        q_mat.append(embedded_token)
    return np.array(q_mat), q_hashes
def score_doc(encoder,hash_size, batch, doc_id):
    q_mat = None
    q_hashes = []
    for token in batch:
        try:
            embedded_token = np.array([encoder[token]])
        except:
            continue
        q_hashes.append(retriever.utils.hash(token, hash_size))
        embedded_token = embedded_token/np.linalg.norm(embedded_token)
        if q_mat is None:
            q_mat = embedded_token
        else:
            q_mat = np.vstack((q_mat,embedded_token))
    c_mat = get_doc_matrix(encoder, doc_id)
    if c_mat is None:
        row, col, data = [], [], []
        return row, col, data 
    else:
        # logging.info(f"cmat: {c_mat.shape}")
        # logging.info(f"q_mat: {q_mat.shape}")
        cosine_sim = np.amax(np.matmul(q_mat, c_mat.T), axis=1)
        assert(cosine_sim.size==len(q_hashes))
        return q_hashes, [doc_id]*len(q_hashes), cosine_sim
def similarity(q_mat, q_hashes, encoder, doc_id):
    c_mat = get_doc_matrix(encoder, doc_id)
    if c_mat is None:
        row, col, data = [], [], []
        return row, col, data 
    else:
        # logging.info(f"cmat: {c_mat.shape}")
        # logging.info(f"q_mat: {q_mat.shape}")
        cosine_sim = np.amax(np.matmul(q_mat, c_mat.T), axis=1)
        assert(len(q_hashes)==q_mat.shape[0])
        logger.info(f"q_hash len: {len(q_hashes)}")
        logger.info(f"c_mat shape: {c_mat.shape}")
        logger.info(f"q_mat shape: {q_mat.shape}")
        logger.info(f"cosim shape: {cosine_sim.size}")
        logger.info(f"cosim shape: {cosine_sim.shape}")
        assert(cosine_sim.size==len(q_hashes))
        return q_hashes, [doc_id]*len(q_hashes), cosine_sim
def get_similarity_matrix(args):
    """Convert the word count matrix into tfidf one.

    matrix:
        X axis = hash(question_word)
        Y axis = doc_id
        value = cosine similarity of word embedding
    """
    logger.info(f"Number of docs: {len(PROCESS_DB.get_doc_ids())}")
    logger.info(f'Loading embedding word vectors {args.embedding}')
    encoder = load_emb(args.embedding)
    
    logger.info(f'Loading question tokens')
    question_tokens = list(get_question_tokens())
    logger.info(f"number of question tokens loaded: {len(question_tokens)}")

    # logger.info(f'Loading corpus tokens')
    # corpus_tokens = get_corpus_tokens(args)
    # logger.info(f"number of corpus tokens: {len(corpus_tokens)}")
    # cosine_sim_data, q_token_hash_row, c_token_hash_col = [], [], []
    logger.info('Constructing question token to doc id alignment sparse matrix')
    doc_ids = PROCESS_DB.get_doc_ids()
    row, col, data = [], [], []
    # step = max(int(len(question_tokens) / 1000), 1)
    # _score = partial(score, encoder, args.hash_size)
    # batches = [question_tokens[i:i + step] for i in range(0, len(question_tokens), step)]
    workers = ProcessPool(
        args.num_workers
    )
    logger.info(f'Loading q_mat for {len(question_tokens)} tokens')
    q_mat, q_hashes = generate_question_token_matrix(encoder, args.hash_size, question_tokens)
    logger.info(f'q_mat has shape: {q_mat.shape}')
    logger.info('Calculating similarity')
    _similarity = partial(similarity, q_mat, q_hashes, encoder)
    count = 0
    with tqdm(total=len(doc_ids)) as pbar:
        # for b_row, b_col, b_data in tqdm(workers.imap_unordered(_similarity, doc_ids)):
        for doc_id in tqdm(doc_ids):
            b_row, b_col, b_data = _similarity(doc_id)
            count += 1
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Storing')
    assert(len(data) == len(row))
    assert(len(row) == len(col))
    matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_ids))
    )
    return matrix


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    
    db_path = "/corpus.db" # subpath to sqlite db in /MedAir/data
    out_dir = "/" # subdirectory in /MedAir/data for saving output files
    embedding_type_default = "bio-wv"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=DATA_DIR + db_path,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('--out_dir', type=str, default=DATA_DIR + out_dir,
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=1,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--embedding', type=str, default=embedding_type_default,
                        help='bio-wv, pubmed-pmc-wv, wiki-pubmed-pmc-wv, or glove')
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='spacy',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    init(tokenizers.get_class(args.tokenizer), retriever.get_class('sqlite'), {'db_path': args.db_path}) #tmp

    # logging.info('Counting words...')
    # count_matrix, doc_dict = get_count_matrix(
    #     args, 'sqlite', {'db_path': args.db_path}
    # )

    logger.info('Making alignment vectors...')
    
    alignment = get_similarity_matrix(args)

    logger.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(count_matrix)

    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += ('-alignment-ngram=%d-embedding=%d-tokenizer=%s' %
                 (args.ngram, embedding_type, args.tokenizer))
    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
        'ngram': args.ngram,
        'doc_dict': doc_dict
    }
    retriever.utils.save_sparse_csr(filename, alignment, metadata)
