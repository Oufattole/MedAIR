from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Q, Search, MultiSearch
from nltk.tokenize import word_tokenize
import query_token

class AIR:
    """
    This class will take a query string containing a question appended with an answer and perform
    the AIR algorithm described in the https://arxiv.org/abs/2005.01218 paper
    """
    def __init__(self, corpus_data, wv):
        self.corpus_data = corpus_data # list of strings
        self.wv = wv # dict from word to vector

    def alignement(self, string_query):
        """
        query: string query
        ignore all query tokens not embedded in self.wv

        Return array of alignment scores for string_query where the index of the score
        cooresponds to the index of the doc from self.corpus_data
        """
        all_query_tokens = set(word_tokenize(string_query)) #set of tokens
        scores = []
        for doc in corpus: # calulate alignment score for all docs
            score = 0
            for word in query_tokens:
                token = query_tokens(token, self.corpus_data, self.wv)
                score += token.align()
            scores.append(score)
        return scores