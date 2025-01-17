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
from gensim.models import KeyedVectors
import pandas as pd
import h5py

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
    wv_from_bin = KeyedVectors.load_word2vec_format(DATA_DIR+"/embeddings/wikipedia-pubmed-and-PMC-w2v.bin", binary=True)  # C bin format
    return wv_from_bin

def load_bio():
    wv_from_bin = KeyedVectors.load_word2vec_format(DATA_DIR+"/embeddings/PubMed-and-PMC-w2v.bin", binary=True)  # C bin format
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

def generate_doc_tensor(embedding, doc_ids):
    # doc_tensor = tf.RaggedTensor
    doc_tensor = None
    # for doc_id in doc_ids:
    #     print(doc_id)
    #     doc_matrix = get_doc_matrix(embedding, doc_id)
        
    #     if not doc_matrix is None:
    #         print(doc_matrix.shape)
    #         doc_matrix = doc_matrix.T
    #         if doc_tensor is None:
    #             doc_tensor = doc_matrix
    #         else:
    #             doc_tensor = np.stack([doc_tensor, doc_matrix], axis=2)
    return doc_tensor


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
def similarity(q_mat,check, encoder, doc_id):
    # global q_mat, q_hashes, encoder
    c_mat = get_doc_matrix(encoder, doc_id)
    if c_mat is None:
        return None, None
    else:
        # logging.info(f"cmat: {c_mat.shape}")
        # logging.info(f"q_mat: {q_mat.shape}")
        cosine_sim = np.amax(np.matmul(q_mat, c_mat.T), axis=1).tolist()
        assert(check==q_mat.shape[0])
        assert(len(cosine_sim)==check)
        return cosine_sim, doc_id
def get_similarity_matrix(args):
    """Convert the word count matrix into tfidf one.

    matrix:
        X axis = hash(question_word)
        Y axis = doc_id
        value = cosine similarity of word embedding
    """
    
    logger.info(f'Loading embedding word vectors {args.embedding}')
    encoder = load_emb(args.embedding)
    
    logger.info(f'Loading question tokens')
    question_tokens = get_question_tokens()

    doc_ids = [i for i in range(len(PROCESS_DB.get_doc_ids()))]

    
    logger.info(f'Loading q_mat for {len(question_tokens)} tokens')
    q_mat, q_hashes = generate_question_token_matrix(encoder, args.hash_size, question_tokens)


    # logger.info(f'Loading doc tensor for {len(doc_ids)} doc_ids')
    # doc_tensor = generate_doc_tensor(encoder, doc_ids)
    # logger.info(f"q_mat: {q_mat.shape} ----- doc_tensor: {doc_tensor.shape}")
    logger.info(f"#docs: {len(doc_ids)} ----- #question_tokens: {len(question_tokens)} ----- #embedded_tokens: {q_mat.shape[0]}")

    # product = np.amax(np.matmul(q_mat, doc_tensor), axis=1)
    # logger.info(f"q_mat: {q_mat.shape} ----- doc_tensor: {doc_tensor.shape} ---- mult: ")
    logger.info("hash_to_ind")
    hash_to_ind = np.array(q_hashes)

    logger.info("allocate array")
    matrix = np.ndarray(shape=(len(q_hashes), len(doc_ids)), dtype=np.float16)

    logger.info("Computing dense matrix")
    _similarity = partial(similarity, q_mat,len(q_hashes), encoder)
    
    for cosine_sim, doc_id in map(_similarity, tqdm(doc_ids)):
        if (not cosine_sim is None):
            assert(len(q_hashes)==len(cosine_sim))
            matrix[:,doc_id] = cosine_sim
    return matrix, hash_to_ind


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
    
    alignment, hash_to_ind = get_similarity_matrix(args)

    
    metadata = {
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
    }
    filename = "word_doc_matrix-"+args.embedding
    logger.info('stored in '+ DATA_DIR +"/table/" + filename)
    retriever.utils.save_dense_array(filename, alignment, hash_to_ind, metadata)
    logger.info('finished')