import nltk
from collections import Counter
from nltk.corpus import stopwords

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


file1 = open("imdb_labelled.txt", "r")
lineNum = 0
for line in file1:
    token = nltk.word_tokenize(line.decode("utf8"))
    lineNum +=1 
    for word in token:
        lookup = index[word.lower()]
        matrix[lineNum-1][lookup]= 1

#matrix with binary equivalent of data
print(matrix)


