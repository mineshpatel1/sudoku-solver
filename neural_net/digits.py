"""
Neural network for learning to recognise 28x28 pixel digits.
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


def conv2d(x, w):
	"""Standard 2D convolutional operation with a stride of 1 and padded with 0s to be the same size as the input."""
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	"""Standard pooling function over 2x2 blocks."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def deep_nn(x):
	"""
	Builds a multi-layered convolutional neural network for classifying digits drawn in 28x28 squares.
	Network contains 5 layers:

		* Input layer: 784 neurons
		* 1st Convolution layer, 5x5 patches of a 28x28 image with 32 outputs. 2x2 max pool downsamples to 14x14 image.
		* 2nd Convolutional layer, 5x5 patches of a 14x14 image with 64 outputs. 2x2 max pool downsamples to 7x7 image.
		* Fully connected layer: 1024 neurons
		* Output layer: 10 neurons

	Total: 38666 neurons

	Args:
		x (tensor): An input tensor with the dimensions (N_examples, 784), where 784 is the number of pixels in a
		standard MNIST image.

	Returns:
		tuple: (y, keep_prob). y is a tensor of shape (N_examples, 10), with values equal to the logits of classifying
		the digit into one of 10 classes (the digits 0-9). keep_prob is a scalar placeholder for the probability of
		dropout.
	"""

	# First convolutional layer that will compute 32 features for each 5x5 patch.
	# Weight variable - first two numbers are the patch size, then the number of input and output channels.
	# A bias variable is defined for each feature as well.
	w_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	# Reshape x into a 4D tensor with width and height in the middle and the last parameter as the number of colour
	# channels.
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	# Convolve x with the weighted tensor and add the bias. Apply the ReLU function.
	# See here for info about ReLU: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
	h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

	# The 2x2 pool layer reduces (downsamples) the 28x28 pixel image to 14x14 pixels
	h_pool1 = max_pool_2x2(h_conv1)

	# Second convolutional layer - in order to build a deep network we stack several layers.
	# Second layer has 64 features for each 5x5 patch
	w_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)  # Reduces image to 7x7 pixels

	# Add a fully connected layer with 1024 neurons to process the entire image
	w_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	# Reshape the pooling layer into a batch of vectors then apply the ReLU function to the new formula
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

	# To reduce overfitting we apply dropout. Use a placeholder so we can apply dropout in training and remove in testing.
	# keep_prob is the probability we will keep the neuron during dropout.
	# See here for info about dropout: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# Finally apply a completion layer, like in the simple example
	w_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

	return y, keep_prob


def test_accuracy(y, y_):
	"""
	Defines method for checking the accuracy of the model.

	Args:
		y (tensor): Predicted values for y in a tensor of shape [N, 10] where N is hte number of samples.
		y_ (tensor): Actual values for y (i.e. the digit classification.

	Returns:
		Accuracy function that can be evaluated against a test set of data.
	"""
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy


def train(data, model_path, test_only=False, steps=1000, batch_size=50, show_test=True):
	"""Trains the neural network in batches of 100, saving the model as it goes

	Args:
		data (Dataset):  Dataset to use, with a train and test set as properties.
		model_path (str): Path to a file to save the model to. Also accepts a previously saved model to pick up from
			where it left off.
		test_only (bool): If set to True, will skip the training phase and only test the data based on the saved `MODEL`.
		steps (int): Number of iterations to train the model for.
		batch_size (int): Number of training elements to use in each batch.
		show_test (bool): Show test accuracy after each training 100 steps as well as the training accuracy. Test accuracy
		 	will always be shown at the end.

	Returns:
		Accuracy of the model as tested on the test set of the input data.
	"""

	x, y_label, y, keep_prob = digit_nn_vars()

	# Define minimisation metric and the training algorithm (Adam optimiser this time)
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	accuracy = test_accuracy(y, y_label)

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
					train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_label: batch[1], keep_prob: 1.0})
					print('Step %d, training accuracy %g' % (i, train_accuracy))
					saver.save(sess, model_path)  # Save model every 100 runs

					if show_test:
						final_accuracy = accuracy.eval(feed_dict={x: data.test.images, y_label: data.test.labels, keep_prob: 1.0})
						print('Test accuracy %g' % final_accuracy)
				train_step.run(feed_dict={x: batch[0], y_label: batch[1], keep_prob: 0.5})

			saver.save(sess, model_path)
		final_accuracy = accuracy.eval(feed_dict={x: data.test.images, y_label: data.test.labels, keep_prob: 1.0})
		print('Test accuracy %g' % final_accuracy)
	return final_accuracy


def predict_digit(test_images, model_path, flatten=True, normalise=True, probabilities=False):
	"""Prediction function

	Args:
		test_images (tensor): Normalised array of pixel values representing a 28x28 digit to classify. Input as a tensor
		of shape [N, 784].
		model_path (str): Path to a pre-saved TensorFlow model.
		flatten (bool): Flattens the image to a one dimensional array of length 784.
		normalise (bool): Normalises the image so each pixel is represented between 0 and 1.
		probabilities (bool): If True, will return the probability array of length 10 for each result instead of the
			prediction.

	Returns:
		str: Classification between 0-9
	"""
	x, y_, y, keep_prob = digit_nn_vars()

	if normalise:
		test_images = [img / 255 for img in test_images]

	if flatten:
		test_images = [img.flatten() for img in test_images]

	# Convert to a Numpy array if necessary
	if type(test_images) == list:
		test_images = np.array(test_images)

	# Initialise the session
	saver = tf.train.Saver()
	with tf.Session() as sess:
		try:
			saver.restore(sess, model_path)
		except:
			print('Could not load saved model.')
			return False

		if probabilities:
			out = y.eval()
		else:
			prediction = tf.argmax(y, 1)
			out = prediction.eval(feed_dict={x: test_images, keep_prob: 1.0})
	return out


def digit_nn_vars():
	"""Initalises variables for the digit Neural Network."""
	# Define variables and shapes
	# x -> 784 is because we're modelling a 28x28 pixel image as a single array
	# y -> 10 as we have 10 classifications, 0 to 9
	# None represents the batch size which can be any
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_label = tf.placeholder(tf.float32, shape=[None, 10])

	y, keep_prob = deep_nn(x)

	# x: Placeholder tensor of shape [N, 784] for the input data.
	# y: Tensor of shape [N, 10] of predicted classification probabilities for the digits (see `deep_nn`).
	# y_: Placeholder tensor of shape [N, 10] for the original classifications of the digits.
	# keep_prob:  Probability to use during dropout (see `deep_nn`).
	return x, y_label, y, keep_prob
