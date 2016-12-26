# encoding=utf-8
'''
Created on 2016-10-17
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
            print '==> ',self.index
            self.index+=1
            sentence='<bos> '+sentence+' <eos>'
            #print sentence
            words = nltk.word_tokenize(sentence)
            #print words,'------------------------------'
            wordNGrams = nltk.ngrams(words, order)
            #print wordNGrams
            for wordNGram in wordNGrams:
                self.ngramFD[wordNGram]+=1
                if order == 1 and wordNGram[0] != "<bos>" and wordNGram[0]!="<eos>":
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
            backoffFreq = self.backoff.ngramFD[ngram[1:]]
            if freq == 0:
                return self.alpha * self.backoff.prob(ngram[1:])
            else:
                return freq / backoffFreq
        else:
            # laplace smoothing to handle unknown unigrams
            return ((self.ngramFD[ngram] + 1) / (self.n + self.v))

def train():
    if os.path.isfile("lm.bin"):
        return
    files = glob.glob("F:/yanjiuyuan/languagemodel/*.txt")
    sentences = []
    i = 0
    for file in files:
        if i > 0 and i % 1 == 0:
            print("%d/%d files loaded, #-sentences: %d" % 
                (i, len(files), len(sentences)))
        #dir, file = file.split("/")
        reader = open(file,'r')
        sentences.extend(re.split(",|\.|\?|;|，|。|？|！|；|\n|\r\n",reader.read()))
        i += 1
    lm = LangModel(3, 0.4, sentences)
    cPickle.dump(lm, open("F:/yanjiuyuan/languagemodel/lm.bin", "wb"))

def test():
    lm1 = cPickle.load(open("lm.bin", 'r'))
    testFile = open("001_jieba.txt", 'r')
    for line in testFile:
        sentence = line.strip()
        sentence='<bos> '+sentence+' <eos>'
            #print sentence
        print "SENTENCE:", sentence,
        words = nltk.word_tokenize(sentence)
        wordTrigrams = nltk.trigrams(words)
        slogprob = 0
        for wordTrigram in wordTrigrams:
            logprob = lm1.logprob(wordTrigram)
            slogprob += logprob
        print "(", slogprob / len(words), ")"

def main():
    #train()
    test()

if __name__ == "__main__":
    main()
