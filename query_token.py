import cosine_similarity from sklearn.metrics.pairwise
import math
import re
from nltk.tokenize import word_tokenize

def findWholeWord(w): # https://stackoverflow.com/questions/5319922/python-check-if-word-is-in-a-string
"""
returns function takes in a string and checks if w is a token in the string
"""
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

coverage_threshold=.95

class Query_Token:
    @classmethod
    def get_IDF(cls, token, text):
        """
        token -- string token 
        corpus -- is a list of strings

        return -- IDF of token in corpus
        """
        doc_frequency = 0
        token_in_doc = findWholeWord(token)
        for doc in corpus:
            if token_in_doc(doc):
                doc_frequency+=1
        total_doc = len(text)
        idf = math.log10((total_doc - doc_frequency + 0.5) / float(doc_frequency + 0.5))
        return idf
    def __init__(self, token, idf, hits, glove_emb):
        """
        token: glove vector for token from query
        idf: idf of token from KB
        hits: list of hits, used to calculate cosine similarity
        """
        self.token_vec = token
        self.idf = idf
        self.covered = False
        self.alignment_score = idf * self.alignment(token_vec, hits)
        self.glove = glove_emb
        

    def alignment(self, token, hits):
        """
        returns sum of max cosine simiarities over all hits
        """
        alignment_scores = []
        for hit in hits:
            max_cosine_similarity = self.align(token, hit)
            if max_cosine_similarity > coverage_threshold:
                this.covered = True
            alignment_scores.append(max_cosine_similarity)
        return sum(alignment_scores)

    def align(self, token, hit):
        """
        returns max cosine similarity
        """
        #we ignore tokens not in glove
        hit_vecs = [self.glove_emb[hit_token[0]] for hit_token in word_tokenize(hit) if hit_token[0] in self.glove_emb]
        token_vec = self.glove_emb[token]
        return max([cosine_similarity(token_vec, hit_vec) for hit_vec in hit_vecs])