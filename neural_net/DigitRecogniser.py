import tensorflow as tf
import numpy as np

import neural_net.digits as nn


def prepare_images(test_images, normalise=True, flatten=True):
	if normalise:
		test_images = [img / 255 for img in test_images]

	if flatten:
		test_images = [img.flatten() for img in test_images]

	# Convert to a Numpy array if necessary
	if type(test_images) == list:
		test_images = np.array(test_images)

	return test_images


class DigitRecogniser:
	def __init__(self, model):
		"""
		Loads the saved model and uses that to predict digits based on input images.

		Args:
			model (str): Path to checkpoint file for the neural network for digit recognition.
		"""
		self.model = model

	def predict_digit(self, test_images, normalise=True, flatten=True, weights=False, threshold=0):
		"""
		Predicts digits from an `np.array` of test_images.

		Args:
			test_images (np.array): Array of test images to predict on. Expects 28x28 size.
			normalise (bool): Normalises the pixel values between 0 and 1.
			flatten (bool): Flattens each image so they are of shape (784) instead of (28, 28)
			weights (bool): Returns the probability weights calculated by the model for each digit.
			threshold (int|float): Minimum weight that should be given to a guess to allow it to be selected.
				Note that this value is not normalised and will be specific to the neural net being used. If the threshold
				is not met, 0 will be used instead of the guessed number.

		Returns:
			np.array: One-dimensional array of predictions for each image provided.
		"""
		test_images = prepare_images(test_images, normalise, flatten)
		tf.reset_default_graph()
		x, y, y_conv, keep_prob = nn.digit_nn_vars()
		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess, self.model)

			if threshold != 0:
				# Weighted likelihood of each image guess
				weights = y_conv.eval(feed_dict={x: test_images, keep_prob: 1.0}, session=sess)
				guesses = np.argmax(weights, 1)  # Guesses for the digits

				# Checks if the guesses are above the threshold weight
				out = []
				for i, guess in enumerate(guesses):
					if weights[i][guess] > threshold:
						out.append(guess)
					else:
						out.append(0)
				out = np.array(out)
			elif weights is True:
				return y_conv.eval(feed_dict={x: test_images, keep_prob: 1.0}, session=sess)
			else:
				prediction = tf.argmax(y_conv, 1)
				out = prediction.eval(feed_dict={x: test_images, keep_prob: 1.0}, session=sess)
		return out

