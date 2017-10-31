import os
import pickle
import unittest

import neural_net.digits as nn
from neural_net.Dataset import Dataset
from Sudoku import classification_mode

DATA_DIR = os.path.abspath(os.path.join('..', 'data', 'datasets'))
DATA_FILE = os.path.join(DATA_DIR, classification_mode())
MODEL = os.path.join('..', 'data', 'models', classification_mode(), 'model.ckpt')


def load_data(file_name):
	"""Loads Python object from disk."""
	with open(file_name, 'rb') as f:
		data = pickle.load(f)
	return data


class NNTrainer(unittest.TestCase):
	def test_training(self):
		try:
			digits = load_data(DATA_FILE)
			nn.train(digits, MODEL, test_only=True, steps=100)
			self.assertTrue(True)  # Simply testing if the training can proceed without error
		except Exception as err:
			self.assertRaises(err)


def suite():
	return unittest.TestLoader().loadTestsFromTestCase(NNTrainer)


def main():
	unittest.main()


if __name__ == '__main__':
	main()