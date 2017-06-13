import nltk
from collections import Counter
from nltk.corpus import stopwords
import theano
import theano.tensor as T
import numpy as np
import random

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

rows = 2000
cols = len(cnt.keys())  #4279
matrix = [ ([0] * cols) for row in range(rows) ]
outputs = [0] * rows


file2 = open("imdb_labelled.txt", "r")
file3 = open("amazon_cells_labelled.txt", "r")
def preproc(file1):
    lineNum = 0
    for line in file2:
        token = nltk.word_tokenize(line.decode("utf8"))
        lineNum +=1 
        y = token[len(token)-1]
        outputs[lineNum-1] = [int(y)]
        token.pop()
        for word in token:
            lookup = index[word.lower()]
            matrix[lineNum-1][lookup]= 1
    for line in file3:
        token = nltk.word_tokenize(line.decode("utf8"))
        lineNum +=1 
        y = token[len(token)-1]
        outputs[lineNum-1] = [int(y)]
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

ranNum = random.sample(range(2000), 200)
for i in range(2000):
    if (i in ranNum): 
        testD.append(array[i])
        testO.append(outputs[i])
    else:
        trainD.append(array[i])
        trainO.append(outputs[i])


X = theano.shared(value=np.asarray(array), name='X')
X1 = theano.shared(value=np.asarray(array), name='X1')
y = theano.shared(value=np.asarray(outputs1), name='y')
rng = np.random.RandomState(1234)
LEARNING_RATE = 0.01


def layer(n_in, n_out):
    return theano.shared(value=np.asarray(rng.uniform(low=-1.0, high=1.0, 
        size=(n_in, n_out)), dtype=theano.config.floatX), name='W', borrow=True)
 
W1 = layer(4279, 7)
W2 = layer(7, 1)


output = T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(X, W1)), W2))
output1 = T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(X1, W1)), W2))
cost = T.sum((y - output) ** 2)
updates = [(W1, W1 - LEARNING_RATE * T.grad(cost, W1)), (W2, W2 - LEARNING_RATE * T.grad(cost, W2))]

train = theano.function(inputs=[], outputs=[], updates=updates)
test = theano.function(inputs=[], outputs=[output1])
 

for i in range(1000):
    if (i+1) % 10 == 0:
        print(i+1)
    train()


new = (test()[0].tolist())
for i in range(2000):
    if (new[i][0] >= 0.5): 
        new[i][0] = 1
    else:
        new[i][0] = 0

counter = 0
for i in range(2000):
    if(new[i][0] == outputs1[i][0]):
        counter= counter +1
print(counter)