"""text search solver"""

import os
from question import Question
import random
from multiprocessing import Pool
import time
import nltk
from nltk import word_tokenize
import jsonlines
from air import AIR
import numpy as np 
import utils
import timeit

class InformationRetrieval():
    """
    runs a query against elasticsearch and sums up the top `topn` scores. by default,
    `topn` is 1, which means it just returns the top score, which is the same behavior as the
    scala solver
    """
    def __init__(self, topn = 50, data_name="dev", output=False, tokenize=False):
        self.title = 0
        print("loading word embedding")
        self.wv=utils.load_glove_emb() # dict word string to word vec
        print("success")
        print("loading corpus")
        self.corpus = utils.load_corpus() # list of string docs
        print("success")
        self.topn = topn
        self.question_filename = data_name+".jsonl"
        self.data_name = data_name
        self.output=output
        if output:
            tokenize_string = "tokenize_" if tokenize else ""
            self.jsonl_filename = "output_"+ tokenize_string +data_name+".jsonl"
            with open(self.jsonl_filename, "w"):
                pass #empty file contents
        self.processes = 1
        self.questions = Question.read_jsonl(self.question_filename)
        # random.shuffle(self.questions)
        self.questions = self.questions[:1]
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
            topn_corpus_indexes, score = self.search_option(prompt, option)
            contexts.append(topn_corpus_indexes)
            option_score[option] = score
        # get answer with highest score
        high_score = max(option_score.values())
        search_answer = None
        for option in option_score:
            if option_score[option]==high_score:
                search_answer = option
        assert(not search_answer is None) # we now have the ir search_answer
        tmp['contexts'] = contexts
        tmp['options'] = options
        tmp['question'] = question.get_prompt()
        tmp['answer_idx'] = question.get_answer_index()
        if self.output:
            with open(self.jsonl_filename, "a") as fp:
                with jsonlines.Writer(fp) as writer:
                    writer.write(tmp)
        return search_answer

    def search_option(self, prompt, option):
        search_string = prompt + " " +option
        if self.tokenize:
            pos_tags_list = nltk.pos_tag(word_tokenize(prompt))
            question_nouns = " ".join([word[0] for word in pos_tags_list
            if word[1] in ["NN", "JJ","NNS","IN"]])
            option_mult = (" " + option)
            search_string = question_nouns + option_mult
        
        ir = AIR(self.corpus_data,self.wv)
        alignment_scores = ir.alignment_scores(search_string)
        topn_indexes = np.argpartition(a, - self.topn)[- self.topn:] # corpus indexes of topn scores
        score = sum([alignment_scores[index] for index in topn_indexes])
        return topn_indexes, score

    def load_question_results(self, responses):
        result = []
        for response in responses:
            result.append(response)
            if len(result) == 4:
                yield result
                result = []

    def answer_all_questions_in_partition(self,questions):
        correct_count = 0
        total = 0
        #Search all queries
        for question in questions:
            search_answer = self.answer_question(question)
            correct_count += 1 if question.is_answer(search_answer) else 0
            total += 1
        return correct_count

    def do_answer_partition(self, i):
        length = len(self.questions)
        interval_length = length//self.processes
        start = interval_length*i 
        end = start+interval_length if i < self.processes-1 else length
        return self.answer_all_questions_in_partition(self.questions[start:end])

    def run(self):
        # pool = Pool(processes=self.processes)
        results = self.answer_all_questions_in_partition(self.questions) # pool.map(self.do_answer_partition, range(0, self.processes))
        tokenize = " tokenize" if self.tokenize else ""
        print(f"{self.data_name + tokenize}; top: {self.topn}; Accuracy: {sum(results)/len(self.questions)}")

def paragraph(topn, data_name,output,tokenize):
    solver = InformationRetrieval(topn=topn, data_name = data_name, output=output,tokenize=tokenize)  # pylint: disable=invalid-name
    print(f"elapsed time: {timeit.timeit(solver.run(),1)}")

if __name__ == "__main__":
    dev = True
    test = False
    output = True
    # tokenize=True
    # paragraph(30,data_name="dev",output=output, tokenize=tokenize)
    # paragraph(30,data_name="test", output=output, tokenize=tokenize)
    # paragraph(30,data_name="train", output=output, tokenize=tokenize)
    tokenize=False
    paragraph(30,data_name="dev",output=output, tokenize=tokenize)
    # paragraph(30,data_name="test", output=output, tokenize=tokenize)
    # paragraph(30,data_name="train", output=output, tokenize=tokenize)
