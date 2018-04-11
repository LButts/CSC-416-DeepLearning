from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labesl, mode):

    #Input layer
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])

    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 25,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool = [2,2], strides = 2)

    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 50,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    # conv3 = tf.layers.conv2d(
    #     inputs = pool2,
    #     filters = 100,
    #     kernel_size = [5,5],
    #     padding = "same",
    #     activation = tf.nn.relu)
    #
    # pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)

    poolflat = tf.reshape(pool2, [-1, 8*8*50])
    fullconn = tf.layers.dense(inputs=poolflat, units=, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=fullconn,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.005)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predicitons["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        
