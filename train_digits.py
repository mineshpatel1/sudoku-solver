import os
import pickle

import neural_net.digits as nn
from neural_net.Dataset import Dataset
from Sudoku import classification_mode


DIGITS = '0123456789'
IMG_DIR = os.path.abspath(os.path.join('data', 'images', 'classified', classification_mode()))
TRAIN_DIR = os.path.join(IMG_DIR, 'train')
TEST_DIR = os.path.join(IMG_DIR, 'test')
DATA_DIR = os.path.abspath(os.path.join('data', 'datasets'))
DATA_FILE = os.path.join(DATA_DIR, classification_mode())
MODEL = os.path.join('data', 'models', classification_mode(), 'model.ckpt')


def save_data(data, file_name):
	"""Saves Python object to disk."""
	with open(file_name, 'wb') as f:
		pickle.dump(data, f)


def load_data(file_name):
	"""Loads Python object from disk."""
	with open(file_name, 'rb') as f:
		data = pickle.load(f)
	return data


def img_labs_from_dir(dir_path):
	"""Gets image paths and labels from the classified image files in a directory."""
	images, labels = [], []
	for digit in DIGITS:
		digit_dir = os.path.join(dir_path, digit)
		files = os.listdir(digit_dir)
		files = list(filter(lambda x: not x.startswith('.'), files))
		files = sorted(files, key=lambda x: int(x.split('.')[0]))

		for file in files:
			images.append(os.path.join(digit_dir, file))
			labels.append(int(digit))
	return images, labels


def create_digit_set(set_name=None, save=True):
	"""
	Creates a dataset with all of the data of classified digits from all the images collected, split into training and
	test and ready for use with Tensorflow. Also saves the file to disk so that the same training/test split can be used
	fairly.
	"""

	img_dir = IMG_DIR
	if set_name is not None:
		img_dir = os.path.abspath(os.path.join('data', 'images', 'classified', set_name))

	# Load in path names from the classified digits folders
	images, labels = img_labs_from_dir(img_dir)

	if save:
		ds = Dataset(images, labels, from_path=True)
		save_data(ds, DATA_FILE)
	else:
		return images, labels


def create_from_train_test(save=True):
	"""Creates a dataset from already divided test and training images."""
	train_images, train_labels = img_labs_from_dir(TRAIN_DIR)
	test_images, test_labels = img_labs_from_dir(TEST_DIR)

	if save:
		ds = Dataset((train_images, test_images), (train_labels, test_labels), from_path=True, split=False)
		save_data(ds, DATA_FILE)
	else:
		return train_images, train_labels, test_images, test_labels


def create_multiple_sets(set_names, combined_name):
	all_images, all_labels = [], []
	for set_ in set_names:
		imgs, labels = create_digit_set(set_name=set_, save=False)
		all_images += imgs
		all_labels += labels
	ds = Dataset(all_images, all_labels, from_path=True)
	save_data(ds, os.path.join(DATA_DIR, combined_name))


def main():
	# create_from_train_test()
	digits = load_data(DATA_FILE)
	nn.train(digits, MODEL, test_only=False, steps=20000, batch_size=50)


if __name__ == '__main__':
	main()
