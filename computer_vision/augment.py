"""
Image augmentation functions to increase the dataset for machine learning purposes. Made redundant by `imgaug`.
"""

import cv2
import numpy as np
import computer_vision.helper as cv


def peturb(img, n=5, background=0):
	"""Shifts image by `n` pixels in a random direction."""
	height, width = img.shape[:2]
	rand = np.random.rand(2)

	def random_triplet(x):
		if 0 <= x < (1/3):
			return -1
		elif (1/3) <= x < (2/3):
			return 0
		else:
			return 1

	horiz = random_triplet(rand[0]) * n
	vert = random_triplet(rand[1]) * n

	m = np.float32([[1, 0, horiz], [0, 1, vert]])
	img = cv2.warpAffine(img, m, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=background)
	return img


def blur(img, max_blur=9):
	"""Implements a randomised Gaussian blur on the image."""
	n = int(max_blur * np.random.rand(1)[0])
	if n % 2 == 0:  # Ensure n is odd
		n += 1
	img = cv2.GaussianBlur(img, (n, n), 0)
	return img


def move(img, n=5):
	"""
	Moves image by n pixels in a random direction, using a background colour that is the mean background of the image.
	"""
	binary = cv2.adaptiveThreshold(img.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)
	background_pixels = [i for i, p in enumerate(binary.flatten()) if p == 255]
	background = int(np.mean((img.flatten()[background_pixels])))

	return peturb(img, n, background)


def move_and_pad(img, move_n=5, pad_n=5):
	"""Moves the image, retaining the dominant background and then pads it with black."""
	img = move(img, move_n)
	return peturb(img, pad_n, background=0)


def contrast(img):
	"""Changes the contrast by a random amount."""
	if np.random.rand(1)[0] > 0.5:
		alpha = 1 + np.random.rand(1)[0]
	else:
		alpha = 1 - np.random.rand(1)[0]

	return cv.adjust_contrast_brightness(img, alpha)


def brightness(img):
	"""Changes the brightness by a random amount."""
	if np.random.rand(1)[0] > 0.5:
		beta = int(255 * np.random.rand(1)[0])
	else:
		beta = int(-255 * np.random.rand(1)[0])

	return cv.adjust_contrast_brightness(img, 1, beta)
