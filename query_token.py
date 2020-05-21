from sklearn.metrics.pairwise import cosine_similarity
import math
import re
from nltk.tokenize import word_tokenize

def findWholeWord(w): # https://stackoverflow.com/questions/5319922/python-check-if-word-is-in-a-string
    """
    returns function takes in a string and checks if w is a token in the string
    """
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def get_IDF(token, corpus):
        """
        token -- string token 
        corpus -- is a list of strings

        return -- IDF of token in corpus
        """
        doc_frequency = 0
        token_in_doc = findWholeWord(token)
        for doc in corpus:
            try:
                if token_in_doc(doc):
                    doc_frequency+=1
            except:
                print('fail')
        total_doc = len(corpus)
        idf = math.log10((total_doc - doc_frequency + 0.5) / float(doc_frequency + 0.5))
        return idf

class Query_Token:
    def __init__(self, token, corpus, wv):
        """
        token: glove vector for token from query
        idf: idf of token from KB
        hits: list of hits, used to calculate cosine similarity
        """
        self.token = token
        self.idf = get_IDF(token, corpus)
        self.covered = False
        self.wv = wv

    def align(self, hit):
        """
        returns max cosine similarity
        """
        token = self.token
        #we ignore tokens not in glove
        hit_vecs = [self.wv[hit_token[0]] for hit_token in word_tokenize(hit) if hit_token[0] in self.wv]
        token_vec = self.wv[token] if token in self.wv else 0
        if len(hit_vecs) and token_vec() != None:
            return max([cosine_similarity(token_vec, hit_vec) for hit_vec in hit_vecs]) * self.idf
        else:
            return 0
