import nltk
from collections import Counter
from nltk.corpus import stopwords
import theano
import theano.tensor as T
import numpy as np
import random
from sklearn import linear_model


logreg = linear_model.LogisticRegression(C=1e5)

#read from text file
file = open("imdb_labelled.txt", "r")
data = file.read()
data = data.decode("utf8")

#tokenize the data
tokens = nltk.word_tokenize(data)

#stop = set(stopwords.words('english'))
#tokens = [i for i in tokens if i not in stop]


#make all words lowercase
for i in range(len(tokens)):
    tokens[i] = tokens[i].lower()
#put into a dictionary
cnt = Counter()
for i in range(len(tokens)):
   word = tokens[i]
   cnt[word]+= 1

#maps each word to an index number
index = dict()
i = 0
for k in cnt.keys():
    index[k] = i
    i+=1


rows = 1000
cols = len(cnt.keys())  #3145
matrix = [ ([0] * cols) for row in range(rows) ]
outputs = [0] * rows


file1 = open("imdb_labelled.txt", "r")
def preproc(file1):
    lineNum = 0
    for line in file1:
        token = nltk.word_tokenize(line.decode("utf8"))
        lineNum +=1 
        y = token[len(token)-1]
        outputs[lineNum-1] = int(y)
        token.pop()
        for word in token:
            lookup = index[word.lower()]
            matrix[lineNum-1][lookup]= 1

    #matrix with binary equivalent of data
    #print(matrix)
    return (matrix, outputs)
    #return outputs
(array, outputs1) = preproc(file1)
testD = list()
testO = list()
trainD = list()
trainO = list()

ranNum = random.sample(range(1000), 100)
for i in range(1000):
    if (i in ranNum): 
        testD.append(array[i])
        testO.append(outputs[i])
    else:
        trainD.append(array[i])
        trainO.append(outputs[i])


logreg.fit(array, outputs1, sample_weight = None)

new = logreg.predict(array)
counter = 0
for i in range(100):
    if(new[i-1] == testO[i-1]):
        counter= counter +1
print(counter)



