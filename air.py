from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Q, Search, MultiSearch
from nltk.tokenize import word_tokenize
import query_token

class AIR:
    """
    This class will take a query string containing a question appended with an answer and perform
    the AIR algorithm described in the https://arxiv.org/abs/2005.01218 paper
    """
    def __init__(self, es, topn, corpus_data, glove_emb):
        self.es = es # elasticsearch client with one index named "corpus" with all the docs in it
        self.topn = topn
        self.corpus_data = corpus_data # list of strings
        self.glove = glove

    def air_retrieval(self, string_query):
        """
        query: string query
        assume all query tokens are embedded with glove

        Return set of all docs retrieved
        """
        hit_set = set()
        full_coverage = False
        previous_query = None
        all_query_tokens = set(word_tokenize(string_query)) #set of tokens
        while(not self.full_coverage and self.previous_query!=self.query): # check stop condition for multi-hop iteration
            es_query = Q('match', body=???)
            search = Search(using=es, index="corpus").query(es_query)[:self.topn]
            hits = search.execute()
            hits = [hit.body for hit in hits]
            for word in query_tokens:
                token_idf = get_idf(word)
                token = query_tokens(token, token_idf, hits, glove_emb)

    def alignement():
        """
        
        """
        pass