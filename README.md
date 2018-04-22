# CSC-416-DeepLearning


This is a convolutional neural net designed to classify images from the
Street View House Numbers (SVHN) dataset. 

It consists of 2 files cnn_svhn.py and svhn_read.py, the data-files are
not included in the repository and are located @ 
http://ufldl.stanford.edu/housenumbers/

To run the neural net just clone the repo into a folder of your choice
and from terminal run $python svhn_read.py
The svhn_read file will download the necessary data-files as well as
process the data to prepare it for the actual neural network

If you already have the files downloaded simply place them in the same
folder as the 2 python files and from terminal run $python svhn_read.py
and it will process the data-files.

Once svhn_read has been run, just run $python cnn_svhn.py and then let
it run. the network will train itself and output some data on loss every
100 iterations or so. My training times averaged about 1.5 - 2 hours.
Once the neural net has finished running it will run the test set and
then output the accuracy of the network for the testing set.