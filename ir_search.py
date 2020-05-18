"""text search solver"""

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Q, Search, MultiSearch
import os
from question import Question
import random
from multiprocessing import Pool
import time
import nltk
from nltk import word_tokenize
import jsonlines
from air import AIR
corpus_data = []
with open("corpus.txt", 'r') as fp:
    corpus_data.extend([each for each in fp.read().splitlines() if len(each)>0]) #get list of Knowledge base docs
embeddings_index = {} #load glove embeddings
f = open("glove.840B.300d.txt",'r', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    # word=lmtzr.lemmatize(word)
    try:
       coefs = np.asarray(values[1:], dtype='float32')
       b = np.linalg.norm(coefs, ord=2)
       coefs=coefs/b
       emb_size=coefs.shape[0]
    except ValueError:
       print (f"glove loading error: {values[0]}")
       continue
    embeddings_index[word] = coefs

es = Elasticsearch() # load elasticsearch client
class InformationRetrieval():
    """
    runs a query against elasticsearch and sums up the top `topn` scores. by default,
    `topn` is 1, which means it just returns the top score, which is the same behavior as the
    scala solver
    """
    def __init__(self, topn = 50, data_name="dev", output=False, tokenize=False):
        self.title = 0
        self.tokenize=tokenize
        self.fp = None
        self.topn = topn
        self.fields = ["body"]
        self.question_filename = data_name+".jsonl"
        self.data_name = data_name
        self.output=output
        if output:
            tokenize_string = "tokenize_" if tokenize else ""
            self.jsonl_filename = "output_"+ tokenize_string +data_name+".jsonl"
            with open(self.jsonl_filename, "w"):
                pass #empty file contents
        self.processes = 8 if not output else 1
        self.questions = Question.read_jsonl(self.question_filename)
        # random.shuffle(self.questions)
        # self.questions = self.questions[:40]
        print(f"Number of Questions loaded: {len(self.questions)}")

    def score(self, hits):
        """get the score from elasticsearch"""
        search_score = sum(hit.meta.score for hit in hits)
        return search_score

    def answer_question(self, question): 
        """
        given a Question object
        returns answer string with the highest score
        """
        tmp = {}
        contexts = [] # list of 4 lists, each containing hits of cooresponding option
        options = question.get_options()
        prompt = question.get_prompt()
        
        option_score = {}
        # get search scores for each answer option
        for i in range(len(options)):
            option = options[i]
            hits = self.search_option(prompt, option)
            contexts.append([hit.body for hit in hits])
        tmp['contexts'] = contexts
        tmp['options'] = options
        tmp['question'] = question.get_prompt()
        tmp['answer_idx'] = question.get_answer_index()
        if self.output:
            with open(self.jsonl_filename, "a") as fp:
                with jsonlines.Writer(fp) as writer:
                    writer.write(tmp)

    def search_option(self, prompt, option):
        search_string = prompt + " " +option
        if self.tokenize:
            pos_tags_list = nltk.pos_tag(word_tokenize(prompt))
            question_nouns = " ".join([word[0] for word in pos_tags_list
            if word[1] in ["NN", "JJ","NNS","IN"]])
            option_mult = (" " + option)
            search_string = question_nouns + option_mult
        info_retriver = AIR(es, self.topn,corpus_data,embeddings_index)
        return info_retriver.air_retrieval(search_string)

    def load_question_results(self, responses):
        result = []
        for response in responses:
            result.append(response)
            if len(result) == 4:
                yield result
                result = []

    def answer_all_questions(self,questions,i):
        correct_count = 0
        total = 0
        #Search all queries
        for question in questions:
            search_answer = self.answer_question(question)
            correct_count += 1 if question.is_answer(search_answer) else 0
            total += 1
        return correct_count

    def do_answer(self, i):
        length = len(self.questions)
        interval_length = length//self.processes
        start = interval_length*i 
        end = start+interval_length if i < self.processes-1 else length
        return self.answer_all_questions(self.questions[start:end], i)
    def run(self):
        start = time.time()
        pool = Pool(processes=self.processes)
        results = pool.map(self.do_answer, range(0, self.processes))
        tokenize = " tokenize" if self.tokenize else ""
        print(f"{self.data_name + tokenize}; top: {self.topn}; Accuracy: {sum(results)/len(self.questions)}")
        print(time.time()-start)

def paragraph(topn, data_name,output,tokenize):
    solver = InformationRetrieval(topn=topn, data_name = data_name, output=output,tokenize=tokenize)  # pylint: disable=invalid-name
    solver.run()

if __name__ == "__main__":
    dev = True
    test = False
    output = True
    tokenize=True
    paragraph(30,data_name="dev",output=output, tokenize=tokenize)
    paragraph(30,data_name="test", output=output, tokenize=tokenize)
    paragraph(30,data_name="train", output=output, tokenize=tokenize)
    tokenize=False
    paragraph(30,data_name="dev",output=output, tokenize=tokenize)
    paragraph(30,data_name="test", output=output, tokenize=tokenize)
    paragraph(30,data_name="train", output=output, tokenize=tokenize)