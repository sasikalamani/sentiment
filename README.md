# sentiment

amazon.py  -  a simple multi layered perceptron 

The amazon data has 1000 sentences with a vocabulary size of 1936. 

Can create multiple layers, change the number of hidden units, and change the learning rate. There is no regularization.

The train set is of size 900 and test of size 100.

To run this file, files needed:
amazon_cells_labelled.txt

run using the command : python amazon.py

AtestItrainLog.py   -   preprocesses the amazon data and the imdb data. The amazon data is used as the test set with a vocabulary size of 1936 and the imdb data is used to train the net. The combined vocabulary size is of 4279. 

Performs a logistic regression to find the accuracy.

To run the file, files needed:
imdb_labelled.txt
amazon_cells_labelled.txt

run using the command : python AtestItrainLog.py

atestitrain.py  - a simple multi layered perceptron. Preprocesses the amazon data and the imdb data. The amazon data is used as the test set with a vocabulary size of 1936 and the imdb data is used to train the net. The combined vocabulary size is of 4279. 

Will print the test error for epochs that see a significant change.
In the end, will print the best validation error and test error.

To run the file, files needed:
imdb_labelled.txt
amazon_cells_labelled.txt

run using the command : python atestitrain.py



logamazimdb.py   -   preprocesses the amazon data and the imdb data. The amazon data has a vocabulary size of 1936. Combined with the imdb dataset, the vocabulary size of 4279. Random split of 1800 for train and 200 for test. 

Performs a logistic regression to find the accuracy.

To run the file, files needed:
imdb_labelled.txt
amazon_cells_labelled.txt

run using the command : python logamazimdb.py



logisticreg.py   -   preprocesses review datasets with 5331 positive reviews and 5331 negative reveiws.

Performs a logistic regression to find the accuracy.

To run the file, files needed:
rt-polarity.neg
rt-polarity.pos

run using the command : python logisticreg.py


MLPamazon.py  -  a simple multi-layered perceptron
add or remove hidden layers with the hidden layer class
tune the hyper parameters with number of hidden units, learning rate, number of epochs, etc. for the yeast dataset

Preprocesses amazon dataset with a vocabulary size of 1936.

Will print the test error for epochs that see a significant change.
In the end, will print the best validation error and test error.

To run the file, files needed:
amazon_cells_labelled.txt
logistic_sgd.py (for the logistic regression layer)

run using the command : python MLPamazon.py


pre.py  -  a simple multi-layered perceptron. Can create multiple layers, change the number of hidden units, and change the learning rate. There is no regularization.

Preprocesses yelp dataset with a vocabulary size of 2088.

Will print the test error for epochs that see a significant change.
In the end, will print the best validation error and test error.

To run the file, files needed:
yelp_labelled.txt
logistic_sgd.py (for the logistic regression layer)

run using the command : python pre.py



rtMLP.py  -  a simple multi-layered perceptron
add or remove hidden layers with the hidden layer class
tune the hyper parameters with number of hidden units, learning rate, number of epochs, etc. for the yeast dataset

Preprocesses review datasets with 5331 positive reviews and 5331 negative reveiws.

Will print the test error for epochs that see a significant change.
In the end, will print the best validation error and test error.

To run the file, files needed:
rt-polarity.neg
rt-polarity.pos
logistic_sgd.py (for the logistic regression layer)

run using the command : python rtMLP.py


lstm.py   -  a simple lstm model tutorial from theano.

Preprocesses amazon dataset with a vocabulary size of 1936.

To run the file, files needed:
amazon_cells_labelled.txt

run using the command : python lstm.py
