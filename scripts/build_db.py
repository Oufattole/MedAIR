#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import logging
import importlib.util

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from alignment.retriever import utils
from alignment import DATA_DIR

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

next_doc_id = 0

# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global next_doc_id
    documents = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = line
            # Skip if it is empty or None
            if not doc or len(doc) == 0:
                continue
            # Add the document
            documents.append((str(next_doc_id), doc, filename))
            next_doc_id += 1
    return documents


def store_contents(data_path, save_path, num_workers=None):
    """store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        num_workers: Number of parallel processes to use when reading docs.
    """
    data_path = DATA_DIR + data_path
    save_path = DATA_DIR + save_path
    if os.path.isfile(save_path):
        os.remove(save_path)
        #raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text, filename);")

    workers = ProcessPool(num_workers)
    files = [f for f in iter_files(data_path) if f[-4:]==".txt"]
    count = 0
    with tqdm(total=len(files)) as pbar:
        for entries in tqdm(workers.imap(get_contents, files)):
            count += len(entries)
            c.executemany("INSERT INTO documents VALUES (?,?,?)", entries)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    data_path = "/corpus"
    save_path = "/corpus.db"
    num_workers=1

    store_contents(
        data_path, save_path, num_workers
    )
