import numpy as np
import cv2


def read_image(path):
	"""Reads image as grayscale and normalises and flattens the array."""
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
	img = img / 255  # Normalise values
	return img.flatten()  # Flatten to shape [784]


def convert_to_one_hot(digit):
	"""Converts an integer between 0-9 into an"""
	if 0 > digit or digit > 9:
		raise ValueError('Only accepts integers between 0 and 9')
	x = [0.] * 10
	x[int(digit)] = 1.
	return x


class Subset:
	def __init__(self, images, labels):
		"""Class for a subset of a dataset, e.g. training or test data."""
		self.num_items = len(images)
		self.images = images
		self.labels = labels
		self.used = []  # Holds all of the indices that have already been used in a previous batch

	def next_batch(self, n):
		"""
		Gets a random batch of images and associated labels to use. Tries not to reuse elements until all elements in
		the set have been used once. This isn't perfect but due to randomisation should still be useful for training.

		Args:
			n (int): Number of items to receive.

		Returns:
			tuple: The first is an `np.array` of randomly selected images and the second is and `np.array` of their
			associated labels.
		"""
		# Reset the used array when it is full
		if len(self.used) + n > self.num_items:
			self.used = np.array([])

		# Restricts to only the unused indices
		possible = range(self.num_items)
		possible = np.array(list(filter(lambda x: x not in self.used, possible)))

		rand = np.random.permutation(len(possible))  # Randomises the order
		batch_idx = rand[:n]  # Takes n elements
		batch_idx = possible[batch_idx]  # Takes the random elements from the list of possible indices to use

		self.used = np.concatenate((self.used, batch_idx), axis=0)
		return self.images[batch_idx], self.labels[batch_idx]


class Dataset:
	def __init__(self, images, labels, from_path=False, normalise=True, flatten=True, one_hot=True, training_ratio=0.8,
	             subsets=False, image_size=28, split=True):
		"""
		Class for holding datasets of images and associated labels. Specifically written for square images of digits.

		Args:
			images (list|tuple): List of images read directly from `cv2`. Expects images to be grayscale.
			labels (list|tuple): List of integers, with each integer indicating the label from 0-9.
			from_path (bool): If True, will load the images from paths given.
			normalise (bool): If True, will normalise the image so each pixel is measured between 0 and 1.
			flatten (bool): If True, will flatten each image array so it is one-dimensional.
			one_hot (bool): If True will convert integer digit labels to lists of length 10 with a 1 indicating the
				position of the value and 0 in each other position.
			training_ratio (float): Indicates the percentage that should be used to randomly split the data into training
				and test segments. E.g. `training=0.8` would mean 80% training data, 20% test.
			subsets (bool): If True, will expect images and labels to be lists or tuples where the first element is
				the array for training data and the second is for test data.
			image_size (int): Length of side of the square image in pixels.
			split (bool): If True, will automatically split the whole dataset according to the `training_ratio`. Otherwise
				will accept tuples for `images` and `labels` where the first element is the training data and the second
				element is the test data.
		"""

		def process(imgs, labs):
			if from_path or normalise or flatten:  # Handles various loading options
				images_ = []
				for img in imgs:
					if from_path:  # Read the image as grayscale
						img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
						if len(img) != image_size ** 2:  # Resize the image if it is not already a square
							img = cv2.resize(img, (28, 28))

					if normalise:  # Normalise the data
						img = img / 255
					if flatten:  # Flatten the array into 1-D
						img = img.flatten()
					images_.append(np.float32(img))
				imgs = images_
				del images_

			if one_hot:
				labs = [convert_to_one_hot(digit) for digit in labs]

			if len(imgs) != len(labs):
				raise AssertionError('Different number of images compared to labels.')

			if type(imgs) is not np.array:
				imgs = np.array(imgs)

			if type(labs) is not np.array:
				labs = np.array(labs)

			return imgs, labs

		if not subsets:
			if split:
				images, labels = process(images, labels)
				self.num_items = len(images)

				# Randomly shuffle the indices and split into a training and test set based on the input ratio
				indices = np.random.permutation(images.shape[0])
				ratio_idx = int(training_ratio * self.num_items)
				training_idx, test_idx = indices[:ratio_idx], indices[ratio_idx:]

				# Creates subsets based on the training and test split
				self.train = Subset(images[training_idx], labels[training_idx])
				self.test = Subset(images[test_idx], labels[test_idx])
			else:
				self.train = Subset(*process(images[0], labels[0]))
				self.test = Subset(*process(images[1], labels[1]))
		else:
			self.train = Subset(images[0], labels[0])
			self.test = Subset(images[1], labels[1])

	@property
	def train_size(self):
		return len(self.train.images)

	@property
	def test_size(self):
		return len(self.test.images)
