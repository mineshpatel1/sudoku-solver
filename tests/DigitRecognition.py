import os
import cv2
import unittest

from neural_net.DigitRecogniser import DigitRecogniser
from Sudoku import classification_mode

IMG_DIR = os.path.join('..', 'data', 'images', 'classified', classification_mode(), 'test')
DIGIT_MODEL = os.path.join('..', 'data', 'models', classification_mode(), 'model.ckpt')
TOLERANCE = 0.98  # Percentage of successful recognition to constitute a pass
DIGIT_RECOGNISER = DigitRecogniser(DIGIT_MODEL)


def test_all_digits(digit):
    """Tests all images of a digit in the classified directory using the neural network for recognising digits."""
    digit_dir = os.path.join(IMG_DIR, str(digit))
    all_images = []
    for img_path in os.listdir(digit_dir):
        img = cv2.imread(os.path.join(digit_dir, img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:  # Skips any hidden files that are not images

            # Resize image if it isn't square
            height, width = img.shape[:2]
            if height != 28 or width != 28:
                img = cv2.resize(img, (28, 28))

            all_images.append(img)
    guesses = DIGIT_RECOGNISER.predict_digit(all_images)
    wrong = len(list(filter(lambda x: int(x) != int(digit), guesses)))
    errors = wrong / len(all_images)
    return errors


class DigitRecognition(unittest.TestCase):
    def test_1(self):
        errors = test_all_digits(1)
        self.assertLessEqual(errors, 1-TOLERANCE)

    def test_2(self):
        errors = test_all_digits(2)
        self.assertLessEqual(errors, 1 - TOLERANCE)

    def test_3(self):
        errors = test_all_digits(3)
        self.assertLessEqual(errors, 1 - TOLERANCE)

    def test_4(self):
        errors = test_all_digits(4)
        self.assertLessEqual(errors, 1 - TOLERANCE)

    def test_5(self):
        errors = test_all_digits(5)
        self.assertLessEqual(errors, 1 - TOLERANCE)

    def test_6(self):
        errors = test_all_digits(6)
        self.assertLessEqual(errors, 1 - TOLERANCE)

    def test_7(self):
        errors = test_all_digits(7)
        self.assertLessEqual(errors, 1 - TOLERANCE)

    def test_8(self):
        errors = test_all_digits(8)
        self.assertLessEqual(errors, 1 - TOLERANCE)

    def test_9(self):
        errors = test_all_digits(9)
        self.assertLessEqual(errors, 1 - TOLERANCE)

    def test_0(self):
        errors = test_all_digits(0)
        self.assertLessEqual(errors, 1 - TOLERANCE)


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(DigitRecognition)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
