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
filei=open("zmk_cn_word_jieba4.txt","r")
ftrain=open("ngramtrain","w")
ftest=open("ngramtest","w")
index=1
for line in filei:
    if index%3==0:
        ftest.write(line)
    else:
        ftrain.write(line)
    index+=1
    print index
filei.close()
ftrain.close()
ftest.close()
