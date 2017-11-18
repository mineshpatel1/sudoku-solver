import os
import cv2
import time
import numpy as np
from shutil import copyfile
from imgaug import augmenters as iaa

import solver
import computer_vision.augment as augment_fns
import computer_vision.helper as cv
from Sudoku import classification_mode
from Sudoku import Sudoku

IMAGE_DIR = os.path.join('data', 'images')
STAGE_DIR = os.path.abspath(os.path.join(IMAGE_DIR, 'stage'))
CLASSIFIED_DIR = os.path.abspath(os.path.join(IMAGE_DIR, 'classified', classification_mode()))
DIGITS_DIR = os.path.abspath(os.path.join(IMAGE_DIR, 'raw', 'digits'))
GRID_DIR = os.path.abspath(os.path.join(IMAGE_DIR, 'grid', 'all'))
TRAIN_DIR = os.path.abspath(os.path.join(IMAGE_DIR, 'grid', 'train'))
TEST_DIR = os.path.abspath(os.path.join(IMAGE_DIR, 'grid', 'test'))
DIGIT_MODEL = os.path.join('data', 'models', classification_mode(), 'model.ckpt')
BEST_MODEL = os.path.join('data', 'best-model', 'model.ckpt')


def mkdir(dir_):
	if not os.path.exists(dir_):
		os.mkdir(dir_)
	return dir_


def get_next_file(name, digit):
	"""Gets the maximum file number in the directory, assuming all filenames are numeric (except for the extension."""
	fs = os.listdir(os.path.join(CLASSIFIED_DIR, name, str(digit)))

	# Ignore any hidden files in the directory
	fs = list(filter(lambda x: not x.startswith('.'), fs))

	if len(fs) > 0:
		return max([int(os.path.basename(f).split('.')[0]) for f in fs]) + 1
	else:
		return 0


def auto_classify(train_only=False, test_only=False, imgaug_seq=None, aug_fn=None, aug_args=(), dry=True, show=False):
	"""
	Automatically "classifies" digit images using the `.dat` file for an image that describes the board. Images are saved
	in the `data/classified/<sub>` directory, with the subdirectory defined by `classification_mode` and `config.ini`
	Can be used to quickly produce new image sets for training and test datasets.

	Args:
		train_only (bool): Only add data to the training set, not the test set.
		test_only (bool): Only add data to the test set, not the training set.
		imgaug_seq (imgaug.Sequential): Post processing function for the digit, can be used to augment datasets.
			Uses the `imgaug` package.
		aug_fn (function): Custom augmentation independent of the `imgaug` library. Mutually exclusive to `imgaug_seq`.
		aug_args(tuple): Arguments to pass to the `aug_fn`.
		dry (bool): If True, won't copy the files to the classified directory.
		show (bool): If True, will show each digit on the screen.
	"""
	if imgaug_seq is not None and aug_fn is not None:
		raise ValueError('Only one of imgaug_seq and aug_fn can be specified.')

	if not os.path.exists(CLASSIFIED_DIR):
		os.mkdir(CLASSIFIED_DIR)

	blank = cv.create_blank_image(28, 28, grayscale=True, include_gray_channel=True)

	def classify_digits(name, src):
		print('%s Classification' % name)

		# Some housekeeping
		if not os.path.exists(os.path.join(CLASSIFIED_DIR, name)):
			os.mkdir(os.path.join(CLASSIFIED_DIR, name))

		for i in range(10):
			digit_dir = os.path.join(CLASSIFIED_DIR, name, str(i))
			if not os.path.exists(digit_dir):
				os.mkdir(digit_dir)

		# Sort files by their number otherwise we'll run into problems when classifying the digits
		files = [f for f in os.listdir(src) if f.split('.')[1] == 'jpg']
		files = sorted(files, key=lambda x: int(x.split('.')[0]))
		for i, f in enumerate(files):
			print('Classifying %s...' % i)
			original = [v.replace('.', '0') for k, v in read_original_board(i, src).items()]
			grid = Sudoku(os.path.join(src, f), include_gray_channel=True)

			# Ignore completely blank images, not required in the training set
			digits_idx = [(j, digit) for j, digit in enumerate(grid.digits) if not np.array_equal(digit, blank)]

			# Modify the image so we can augment the training set for better neural networks
			# Structured to use `imgaug.Sequential` objects to describe transformations
			if imgaug_seq is not None:
				digits = [x[1] for x in digits_idx]
				digits = imgaug_seq.augment_images(digits)
				digits_idx = [(digits_idx[k][0], digit) for k, digit in enumerate(digits)]

			# Modify the image using a custom function outside of the imgaug library
			if aug_fn is not None:
				digits_idx = [(d[0], aug_fn(d[1], *aug_args)) for d in digits_idx]

			for j, digit in digits_idx:
				if show:
					cv.show_image(digit)
				if not dry:  # If not a dry run, write the image to the relevant directory
					# Only keep a small percentage of blank cells when augmenting to avoid huge skewing of dataset.
					# Also should allow us to build larger, meaningful sets and train the models faster.
					if original[j] == '0' and (imgaug_seq is not None or aug_fn is not None):
						if np.random.randint(50) != 0:
							continue

					cv2.imwrite(os.path.join(CLASSIFIED_DIR, name, original[j], '%s.jpg' % get_next_file(name, original[j])),
					            digit)

	if not test_only:
		classify_digits('train', TRAIN_DIR)
	if not train_only:
		classify_digits('test', TEST_DIR)


def parse_grid(idx, model_path=None, save_board=False):
	"""
	Parses a Sudoku image from the grid directory and saves the digits and squares to the appropriate raw directories.
	"""
	image_path = os.path.join(GRID_DIR, '%s.jpg' % idx)
	grid = Sudoku(image_path, model_path)

	if save_board:
		data_file = os.path.join(GRID_DIR, '%s.dat' % idx)
		with open(data_file, 'w') as f:
			f.write(grid.board)

	return grid


def read_original_board(sudoku_id, dir_path=None, as_string=False, as_list=False):
	"""Reads the .dat file with the original board layout recorded."""
	folder = GRID_DIR
	if dir_path is not None:
		folder = dir_path

	with open(os.path.join(folder, '%s.dat' % sudoku_id), 'r') as f:
		original = f.read()

	if as_string:
		return original
	elif as_list:
		return [v for k, v in solver.parse_sudoku_puzzle(original).items()]
	else:
		return solver.parse_sudoku_puzzle(original)


def test_board_recognition(all_grids=True, model=None):
	"""
	Runs the Sudoku processing for each of the IDs submitted.


	Args:
		all_grids (bool): If True, tests all grids, otherwise just tests against the test set, of boards and digits the model
			has never seen.
		model (str): Path to the model that should be used for evaluation. If None, will use the best model saved.

	Returns:
		list: List of result records, with the following properties:

		* `success`: a boolean corresponding to the whether the board was successfully
		* `id`: ID of the run.
		* `img_path`: Path to the image of the Sudoku board.
		* `elapsed`: Time in seconds taken for processing.
		* `diff`: Dictionary of dictionaries, with the `original` board and the `guess`, filtered for positions with
			incorrect guesses.
		* `num_diff`: Number of incorrectly recognised squares.
	"""
	results = []

	def get_files(dir_path):
		return sorted([f for f in os.listdir(dir_path) if f.split('.')[1] == 'jpg'], key=lambda f: int(f.split('.')[0]))

	if all_grids:
		img_dir = GRID_DIR
	else:
		img_dir = TEST_DIR

	for i, f in enumerate(get_files(img_dir)):
		print('Processing %s...' % f)
		original = read_original_board(i, img_dir)
		img = os.path.join(img_dir, f)

		start = time.time()
		try:
			sudoku = Sudoku(img, model)
			result = {'id': i, 'img_path': img, 'elapsed': time.time() - start}
			guess = sudoku.board_dict

			diff = []
			for position in guess:
				if guess[position] != original[position]:
					diff.append(position)

			result['num_diff'] = len(diff)
			result['diff'] = {
				'original': {k: v for k, v in original.items() if k in diff},
				'guess': {k: v for k, v in guess.items() if k in diff}
			}
		except:
			result = {'id': i, 'img_path': img, 'elapsed': time.time() - start, 'num_diff': 81}
			guess = False

		if original == guess:
			result['success'] = True
		else:
			result['success'] = False
		results.append(result)

	success = len(list(filter(lambda x: x['success'], results)))
	failures = [x['id'] for x in results if not x['success']]
	success_ratio = success / len(results)
	total_time = sum([x['elapsed'] for x in results])
	avg_time = total_time / len(results)
	total_digits = 81 * len(results)
	wrong_digits = sum([x['num_diff'] for x in results])
	print('Success: %s\tFail: %s\tRatio: %s' % (success, len(results) - success, success_ratio))
	print('Elapsed: %ss\tAverage: %ss' % (total_time, avg_time))
	print('Total Digits: %s\tMissed Digits: %s\tRatio: %s' % (total_digits, wrong_digits, wrong_digits / total_digits))
	print('Failures:')
	[print('%s' % x) for x in failures]
	return results


def save_failed_digits(results, model=None):
	"""Saves the images of the failed digits and prints their guessed values."""
	if model is None:  # Default to the best model
		model = os.path.join(os.path.dirname(__file__), 'data', 'best-model', 'model.ckpt')

	failed_dir = os.path.join(os.path.dirname(model), 'failed_digits')
	mkdir(failed_dir)
	for result in results:
		if result['num_diff'] > 0:
			bad_digits = []
			example = parse_grid(result['id'], model)
			original = read_original_board(result['id'], GRID_DIR, as_list=True)
			print('\nBoard %s:' % result['id'])
			print('\t\tGuess\tActual')
			for guess in result['diff']['guess']:
				print('%s\t\t%s\t\t%s' % (guess, example.board_int[solver.coord_to_idx(guess)], original[solver.coord_to_idx(guess)]))
				bad_digits.append(example.digits[solver.coord_to_idx(guess)])
			cv2.imwrite(os.path.join(failed_dir, ('%s.jpg' % result['id'])), np.concatenate(np.array(bad_digits), axis=1))


def check_for_duplicates(ids):
	boards = {}
	for unique_id, i in enumerate(ids):
		b = read_original_board(i, as_string=True)
		boards[b] = boards.get(b, {})
		boards[b]['unique_id'] = unique_id
		boards[b]['ids'] = boards[b].get('ids', [])
		boards[b]['ids'].append(i)
		boards[b]['count'] = boards[b].get('count', 0) + 1
	boards = {v['unique_id']: {'count': v['count'], 'ids': v['ids']} for k, v in boards.items() if v['count'] > 1}
	print(boards)


def random_train_test(num_train=50):
	grids = [f.split('.')[0] for f in os.listdir(GRID_DIR) if f.split('.')[1] == 'dat' and not f.startswith('.')]
	rand = np.random.permutation(len(grids))
	train_idx, test_idx = rand[:num_train], rand[num_train:]

	def copy_subset(indices, dir_path):
		for f in os.listdir(dir_path):  # Clear out current directory
			os.unlink(os.path.join(dir_path, f))

		for i, idx in enumerate(np.array(grids)[indices]):
			copyfile(os.path.join(GRID_DIR, '%s.jpg' % idx), os.path.join(dir_path, '%s.jpg' % i))
			copyfile(os.path.join(GRID_DIR, '%s.dat' % idx), os.path.join(dir_path, '%s.dat' % i))

	copy_subset(train_idx, TRAIN_DIR)
	copy_subset(test_idx, TEST_DIR)


def main():
	model = None  # None chooses the current best model

	# Augmentation transformations
	# Can play around with these and massively expand the training set
	# seq = iaa.Sequential([
		# iaa.Crop(px=(2, 5)),  # Crop images from each side by 0 to 4px (randomly directions)
		# iaa.Pad(px=(0, 4)),  # Pad images for each side by 0 to 4px (random directions)
		# iaa.GaussianBlur(sigma=0.5),  # Gaussian blur with a kernel of size sigma
		# iaa.Dropout(p=0.2),  # Adds randomised pixel dropout
		# iaa.PerspectiveTransform(scale=0.075, keep_size=True)
	# ])

	# auto_classify(train_only=True, dry=False)  # Standard extraction
	# auto_classify(train_only=True, imgaug_seq=seq, dry=False)  # Manipulation using imgaug
	# auto_classify(train_only=True, aug_fn=augment_fns.contrast, dry=False)  # Custom manipulation

	results = test_board_recognition(True)
	# save_failed_digits(results, model))


if __name__ == '__main__':
	main()
