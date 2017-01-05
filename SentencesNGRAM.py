# encoding=utf-8
'''
Created on 2017-1-5
@author: user
'''
from __future__ import division

import math
import os.path
import jieba
import cPickle
import glob
import nltk
import re

class LangModel:
    def __init__(self, order, alpha, sentences):
        self.index=0
        
        self.order = order
        self.alpha = alpha
        if order > 1:
            self.backoff = LangModel(order - 1, alpha, sentences)
            self.lexicon = None
        else:
            self.backoff = None
            self.n = 0
        self.ngramFD = nltk.FreqDist()
        lexicon = set()
        for sentence in sentences:
            #print '<bos> '+sentence+' <eos>'
            #print '==> ',self.index
            self.index+=1
            #sentence='<bos> '+sentence+' <eos>'
            #sentence=sentence.decode("utf8")
            #print sentence
            words = sentence.split(" ")
            #print words,'------------------------------'
            wordNGrams = nltk.ngrams(words, order)
            #print wordNGrams
            for wordNGram in wordNGrams:
                #print wordNGram
                self.ngramFD[wordNGram]+=1
                if order == 1 and wordNGram[0] != "<s>" and wordNGram[0]!="</s>":
                    lexicon.add(wordNGram)
                    self.n += 1
        self.v = len(lexicon)
        
#         for k,v in self.ngramFD.items():
#             print k, v,'==================='

    def logprob(self, ngram):
        return math.log(self.prob(ngram))
    
    def prob(self, ngram):
        if self.backoff != None:
            
            freq = self.ngramFD[ngram]
            backoffFreq = self.backoff.ngramFD[ngram[:-1]]
            #print ngram,ngram[:-1],"=========="
            if freq == 0:
                backprob=self.backoff.prob(ngram[1:])
                return self.alpha * backprob
            else:
                return freq / backoffFreq
        else:
            # laplace smoothing to handle unknown unigrams
            print self.n,"-=-=-=-=-"
            return ((self.ngramFD[ngram] + 1) / (self.n + self.v))

def train():
    if os.path.isfile("lm.bin"):
        return
    files = glob.glob("F:/yanjiuyuan/languagemodel/test/*.txt")
    sentences = []
    i = 0
    for file in files:
        if i > 0 and i % 10000 == 0:
            print("%d/%d files loaded, #-sentences: %d" % 
                (i, len(files), len(sentences)))
        #dir, file = file.split("/")
        reader = open(file,'r')
        sentences.extend(re.split("\n",reader.read()))
        i += 1
    lm = LangModel(3, 0.4, sentences)
    cPickle.dump(lm, open("F:/yanjiuyuan/languagemodel/test/lm.bin", "wb"))

def test():
    lm1 = cPickle.load(open("F:/yanjiuyuan/languagemodel/test/lm.bin", 'r'))
    testFile = open("F:/yanjiuyuan/languagemodel/test/totest.txt1", 'r')
    for line in testFile:
        sentence = line.strip()
            #print sentence
        print "SENTENCE:", sentence
        words = sentence.split(" ")
        wordngrams = nltk.ngrams(words, len(words))
        slogprob = 0
        for wordTrigram in wordngrams:
            logprob = lm1.logprob(wordTrigram)
            slogprob += logprob
        print "(", slogprob / len(words), ")"
def get_log_prob(sentence,lm=None):
    
    words = sentence.split(" ")
    if(len(words)!=3):
        raise Exception("Invalid ngram !")
    wordngrams = nltk.ngrams(words, len(words))
    for wordTrigram in wordngrams:
        logprob = lm.logprob(wordTrigram)
        return logprob

def get_raw_prob(sentence,lm=None):
    
    words = sentence.split(" ")
    if(len(words)!=3):
        raise Exception("Invalid ngram !")
    wordngrams = nltk.ngrams(words, len(words))
    for wordTrigram in wordngrams:
        prob = lm.prob(wordTrigram)
        return prob
    

def main():
    lm1 = cPickle.load(open("F:/yanjiuyuan/languagemodel/libo_baseline/lm.bin", 'r'))
    print get_log_prob("啊 到 这", lm1)
    print get_raw_prob("啊 到 这", lm1)
    #train()
    #test()
    

if __name__ == "__main__":
    main()
