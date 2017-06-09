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

print(len(cnt.keys()))



