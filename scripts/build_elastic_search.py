#!/usr/bin/env python
from __future__ import print_function

import json
import re
import sys
import os
import requests
from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Index
from tqdm import tqdm

from alignment import DATA_DIR

# issues bulk insert commands to an Elasticsearch server running
# on localhost.
es = Elasticsearch()
def sentences_to_id_doc(sentences, filename):
    sentence_id = 0
    valid = []
    for sentence in sentences:
        yield {"body":sentence, "_id":sentence_id}
        sentence_id += 1
def bulk_load_elasticsearch(sentences, filename):
    print(f"loading {filename}")
    sentence_generator = sentences_to_id_doc(sentences, filename)
    bulk_sender = helpers.parallel_bulk(es, tqdm(sentence_generator, total=len(sentences)), index = "corpus")
    for success, info in bulk_sender:
        if not success:
            print('A document failed:', info)

def txt_to_paragraphs(data):
    sentences=[]
    for line in data:
        if len(line)==0 or line=="\n":
            continue
        sentences.append(line)
    assert(len(sentences)== 231581)
    return sentences

def group_paragraphs(filename):
    paragraphs = None
    with open(filename, 'r') as fp:
        paragraphs = txt_to_paragraphs(fp)
    return paragraphs

def load_paragraphs():
    filename = DATA_DIR + "/corpus/corpus.txt"
    delete_search_indexes()
    sentences = group_paragraphs(filename)
    bulk_load_elasticsearch(sentences, filename)
    os.chdir('..')

def delete_search_indexes():
    es.indices.delete(index=["*.txt","corpus"], ignore=[400, 404])
    
if __name__ == "__main__":
    load_paragraphs()
    