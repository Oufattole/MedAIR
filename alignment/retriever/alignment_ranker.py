#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""

import logging
import numpy as np
import scipy.sparse as sp

from multiprocessing import Pool, RawArray
import ctypes as c
from functools import partial
import h5py
from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Q, Search, MultiSearch
from tqdm import tqdm

import utils
from alignment import tokenizers
from question import Question
from alignment import DATA_DIR

tracer = logging.getLogger('elasticsearch')
tracer.setLevel(logging.CRITICAL) # or desired level
tracer.addHandler(logging.FileHandler('indexer.log'))


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

shared_matrix = None
shared_matrix_shape = None


def init(matrix, shape):
    global shared_matrix, shared_matrix_shape
    shared_matrix = matrix
    shared_matrix_shape = shape

class AlignmentDocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, embedding):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        self.num_workers=1
        self.embedding = embedding
        alignment_filename = "word_doc_matrix-" + embedding
        logger.info(f'Loading {embedding} matrix')
        matrix, hash_to_ind, metadata = utils.load_dense_array(alignment_filename)
        self.hash_to_ind = {hash_to_ind[i]:i for i in range(hash_to_ind.size)}
        # import pdb; pdb.set_trace()
        logger.info(f'Allocate RawArray')
        self.matrix = RawArray(c.c_double, matrix.shape[0]*matrix.shape[1])
        self.matrix_shape = matrix.shape
        logger.info(f'make wrapper')
        matrix_wrapper = np.frombuffer(self.matrix).reshape(matrix.shape)
        logger.info(f'copy to RawArray')
        np.copyto(matrix_wrapper, matrix[:,:])
        # self.matrix = matrix
        tokenizer_type = metadata["tokenizer"]
        self.tokenizer = tokenizers.get_class(tokenizer_type)()
        self.hash_size = metadata["hash_size"]

        basename = ('-wordDocFreq-hash=%d-tokenizer=%s.npz' %
                    (self.hash_size, tokenizer_type))
        logger.info(f'Loading word frequency table')
        word_freq_filename = DATA_DIR +"/corpus"+ basename
        self.freq_table = utils.load_word_freq(word_freq_filename)
        self.es_topn = 1000
        self.topn = 30
        self.es = Elasticsearch()


    def solve_question_set(self, questions):
        total = 0
        correct = 0 
        pbar = tqdm(questions, desc='accuracy', total = len(questions))
        with Pool(processes=self.num_workers,initializer=init,initargs=(self.matrix,self.matrix_shape)) as pool:
            _solve = partial(solve_question, self.hash_to_ind, self.freq_table, self.tokenizer, self.es, self.es_topn, self.topn,self.hash_size)
            for result in pool.imap(_solve, pbar):
                total +=1
                correct += 1 if result else 0
                pbar.set_description(f'accuracy (accuracy={correct/total})')
                pbar.update()

        return correct/total
    def solve_dev_set(self):
        logger.info(f'Solving dev set for {self.embedding}')
        question_filename = "/questions/dev.jsonl"
        questions = Question.read_jsonl(DATA_DIR + question_filename)
        return self.solve_question_set(questions)
    def solve_test_set(self):
        logger.info(f'Solving test set for {self.embedding}')
        question_filename = "/questions/test.jsonl"
        questions = Question.read_jsonl(DATA_DIR + question_filename)
        return self.solve_question_set(questions)
    def solve_train_set(self):
        logger.info(f'Solving training set for {self.embedding}')
        question_filename = "/questions/train.jsonl"
        questions = Question.read_jsonl(DATA_DIR + question_filename)
        return self.solve_question_set(questions)

def solve_question(hash_to_ind, freq_table, tokenizer, es, es_topn, topn, hash_size, question_object):
    options = list(question_object.get_options())
    queries = [question_object.prompt + " " + option for option in options]
    _solve = partial(score_query, hash_to_ind, freq_table, tokenizer, es, es_topn, topn, hash_size)
    scores = [_solve(query) for query in queries]
    high_score = max(scores)
    search_answer = None
    for i in range(len(scores)):
        if scores[i]==high_score:
            search_answer = options[i]
    return question_object.is_answer(search_answer)

def score_query(hash_to_ind, freq_table, tokenizer, es, es_topn, topn, hash_size, input_query):
    """
    query : question + answer

    returns alignment score


    """
    # logger.debug(f'es_search')
    global shared_matrix
    global shared_matrix_shape
    matrix = np.frombuffer(shared_matrix).reshape(shared_matrix_shape)
    # tokenizer = self.tokenizer
    # es = self.es
    #formulate search query
    es_query = Q('match', body=input_query)
    search = Search(using=es, index="corpus").query(es_query).source(False)[:es_topn]
    hits = search.execute()
    doc_ids = np.array([int(hit.meta.id) for hit in hits])
    doc_ids.sort()
    tokens = list(tokenizer.tokenize(input_query).words()) # globalize tokenize with init
    valid_hashes = [utils.hash(token, hash_size) for token in tokens if utils.hash(token, hash_size) in hash_to_ind]
    valid_hash_inds = np.array([hash_to_ind[token_hash] for token_hash in valid_hashes])
    valid_hash_inds.sort()
    # retrieve matrix
    cos_sim_matrix = matrix[valid_hash_inds[:,None],doc_ids]
    # logger.debug(f'retrieve up matrix')
    # cos_sim_matrix = matrix[:,doc_ids][valid_hash_inds,:]
    # retrieve idfs
    # logger.debug(f'Calculate the rest')
    num_docs = matrix.shape[1]
    Ns = np.array([freq_table[token_hash] for token_hash in valid_hashes]) #doc frequencies array same order as 
    idfs = np.log((num_docs - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0

    assert(cos_sim_matrix.shape[0]==idfs.size)
    
    scores = (np.dot(idfs,cos_sim_matrix)) #find correct axis, sum rows
    topn_scores = scores.argsort()[-topn:]
    # logger.info(f'Done')
    return np.sum(scores[topn_scores])


if __name__ == "__main__":
    embeddings = ["bio-wv", "pubmed-pmc-wv", "wiki-pubmed-pmc-wv", "glove"]
    embedding = "bio-wv"
    air = AlignmentDocRanker(embedding)
    x = air.solve_dev_set()
    print(x)
    x = air.solve_test_set()
    print(x)
    x = air.solve_train_set()
    print(x)