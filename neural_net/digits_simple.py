"""
Simple Neural network for learning to recognise 28x28 pixel digits.
Based heavily on TensorFlow's Deep MNIST tutorial: https://www.tensorflow.org/get_started/mnist/pros
"""

import os
import numpy as np
import tensorflow as tf


def weight_variable(shape):
    """
    Defines a weight variable with a small amount of noise to break symmetries and reduce the number of 0 gradients.
    This reduces the amount of "dead" neurons that don't update themselves.
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    Defines a slightly positive bias in a given shape. This is useful to reduce the number of dead neurons.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def simple_nn():
    """
    Builds a simple neural neural network using gradient descent optimisation and backpropagation for
    classifying digits drawn in 28x28 squares. Network contains 3 layers:

    * Input layer: 784 neurons
    * Hidden layer: 16 neurons
    * Output layer: 10 neurons

    Total: 810 neurons


    Args:
        x (tensor): An input tensor with the dimensions (N_examples, 784), where 784 is the number of pixels in a
        standard MNIST image.

    Returns:
        tensor: Tensor of shape (N_examples, 10), with values equal to the logits classifying the digit into one of
            10 classes (digis 0-9).
    """

    x = tf.placeholder(tf.float32, shape=[None, 784])  # Placeholder for input
    y_label = tf.placeholder(tf.float32, shape=[None, 10])  # Placeholder for true labels (used in training)
    hidden_neurons = 16  # Number of neurons in the hidden layer, constant

    # Hidden layer
    w_1 = weight_variable([784, hidden_neurons])
    b_1 = bias_variable([hidden_neurons])
    h_1 = tf.nn.sigmoid(tf.matmul(x, w_1) + b_1)  # Order of x and w_1 matters here purely syntactically

    # Output layer
    w_2 = weight_variable([hidden_neurons, 10])
    b_2 = bias_variable([10])
    y = tf.matmul(h_1, w_2) + b_2  # Note that we don't use sigmoid here because the next step uses softmax

    return x, y_label, y


def train(data, model_path, test_only=False, steps=1000, batch_size=50, show_test=True):
    x, y_label, y = simple_nn()

    # Cross entropy cost function
    # More numerically stable to perform Softmax here instead of on the previous layer
    # c.f. https://www.tensorflow.org/get_started/mnist/beginners
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y))

    # Gradient descent and backpropagation learning
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

    # Accuracy comparison/measurement function
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Creates the model directory if it doesn't already exist
    if not os.path.exists(os.path.dirname(model_path)):
        os.mkdir(os.path.dirname(model_path))

    # Initialise the session
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver.restore(sess, model_path)
            print('Loaded the model.')
        except:
            if test_only:
                return 0

        if not test_only:
            for i in range(steps):
                batch = data.train.next_batch(batch_size)
                if i % 100 == 0:
                    # Check on training accuracy
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_label: batch[1]})
                    print('Step %d, training accuracy %g' % (i, train_accuracy))
                    saver.save(sess, model_path)  # Save model every 100 runs

                    # Check accuracy of the test set (unseen images)
                    if show_test:
                        final_accuracy = accuracy.eval(
                            feed_dict={x: data.test.images, y_label: data.test.labels})
                        print('Test accuracy %g' % final_accuracy)
                train_step.run(feed_dict={x: batch[0], y_label: batch[1]})

            saver.save(sess, model_path)

        # Print final test accuracy
        final_accuracy = accuracy.eval(feed_dict={x: data.test.images, y_label: data.test.labels})
        print('Test accuracy %g' % final_accuracy)
    return final_accuracy
