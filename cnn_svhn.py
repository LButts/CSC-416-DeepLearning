from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from svhn_read import load_data

tf.logging.set_verbosity(tf.logging.INFO)


# Defenition of CNN model
def cnn_model_fn(features, labels, mode):

    #sets the shape of the input layer to match that of the data
	#32-by-32 pixels with 3 color channels
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])

	#set first convolutional layer to have 25 5-by-5 filters
	#with zero-padding and ReLU activation
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 25,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu)

	#use max pooling to reduce the data dimensions by a factor of 2
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 2)

	#set up second convolutional layer, same as the first but w/ 50 filters
	
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 50,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu)

	#use max pooling again to reduce the data dimensions
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

	#convert the last pooled layer into a "flat" 2D matrix
	#for the fully connected layer to use,
	#with dimensions of [num_pics, X*X*Y] where X is the 
	#length of one side of the input and Y is the width
    poolflat = tf.reshape(pool2, [-1, 8*8*50])
	
	#set fully connected layer with 1024 nuerons and an
	#activation function of ReLU
    fullconn = tf.layers.dense(inputs=poolflat, units=1024, activation=tf.nn.relu)
    #sets the dropout rate to 40%
	dropout = tf.layers.dropout(
        inputs=fullconn,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

	#sets labels as possible predictions from logits
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

	
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	#set loss function as sparse_softmax_cross_entropy
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits)

	#Set up training part of model, with learning rate of 0.005
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.005)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

	#set the eval metric to use accuracy.
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(argv):

	#load in training and eval data using load_data
	#from svhn_read
    train_data, train_labels = load_data('train')
    eval_data, eval_labels = load_data('test')

	#instantiate the classifier, sets the tensorboard
	#directory as "/tmp/svhn_concnn_model" for 
	#retreival of tensorboard data
    svhn_classifier = tf.estimator.Estimator(
        model_fn = cnn_model_fn, model_dir="/tmp/svhn_concnn_model")

	#set the log to use probabilities and set up 
	#the loghook to log every 50 iterations
    tensorLog = {"probabilities": "softmax_tensor"}
    logHook = tf.train.LoggingTensorHook(tensors=tensorLog, every_n_iter=50)

	#run the training data with a batch_size of 125 and shuffling
	#runs for 20,000 iterations then runs test set
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size = 125,
        num_epochs=None,
        shuffle=True)
    svhn_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logHook])

	#runs the test set and prints the test results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = svhn_classifier.evaluate(input_fn = eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
