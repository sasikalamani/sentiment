#train with imbd, test with amazon, logistic regression

import nltk
from collections import Counter
from nltk.corpus import stopwords
import theano
import theano.tensor as T
import numpy as np
import random
from sklearn import linear_model

#read from text file
file = open("amazon_cells_labelled.txt", "r")
file1 = open("imdb_labelled.txt", "r")
data1 = file1.read().decode("utf8")
data = file.read().decode("utf8")

#tokenize the data
tokens = nltk.word_tokenize(data)
tokens1 = nltk.word_tokenize(data1)

#make all words lowercase
for i in range(len(tokens)):
    tokens[i] = tokens[i].lower()
for i in range(len(tokens1)):
    tokens1[i] = tokens1[i].lower()
#put into a dictionary
cnt = Counter()
for i in range(len(tokens)):
   word = tokens[i]
   cnt[word]+= 1

for i in range(len(tokens1)):
   word = tokens1[i]
   cnt[word]+= 1

#len cnt 4279

#maps each word to an index number
index = dict()
i = 0
for k in cnt.keys():
    index[k] = i
    i+=1

rows = 1000
cols = len(cnt.keys())  #4279
matrix = [ ([0] * cols) for row in range(rows) ]
outputs = [0] * rows
matrixT = [ ([0] * cols) for row in range(rows) ]
outputsT = [0] * rows


file2 = open("imdb_labelled.txt", "r")
file3 = open("amazon_cells_labelled.txt", "r")
def preproc(file1):
    lineNum = 0
    for line in file2:
        token = nltk.word_tokenize(line.decode("utf8"))
        lineNum +=1 
        y = token[len(token)-1]
        outputs[lineNum-1] = int(y)
        token.pop()
        for word in token:
            lookup = index[word.lower()]
            matrix[lineNum-1][lookup]= 1
    lineNum1 = 0
    for line in file3:
        token = nltk.word_tokenize(line.decode("utf8"))
        lineNum1 +=1 
        y = token[len(token)-1]
        outputsT[lineNum1-1] = int(y)
        token.pop()
        for word in token:
            lookup = index[word.lower()]
            matrixT[lineNum1-1][lookup]= 1

    #matrix with binary equivalent of data
    #print(matrix)
    return (matrix, outputs, matrixT, outputsT)
    #return outputs
(array, outputs1, array2, outputs2) = preproc(file1)

logreg = linear_model.LogisticRegression(dual=False, random_state= 2)
#logreg.fit(trainD, trainO, sample_weight = None)
logreg.fit(array, outputs1, sample_weight = None)

#new = logreg.predict(testD)
new = logreg.predict(array2)

counter = 0
for i in range(1000):
    #if(new[i-1] == testO[i-1]):
    if(new[i-1] == outputs2[i-1]):
        counter= counter +1
print(counter)
