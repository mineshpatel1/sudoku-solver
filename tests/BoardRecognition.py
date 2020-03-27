import os
import time
import solver
import unittest

from Sudoku import Sudoku
from Sudoku import classification_mode

GRID_DIR = os.path.join('..', 'data', 'images', 'grid', 'all')
DIGIT_MODEL = os.path.join('..', 'data', 'models', classification_mode(), 'model.ckpt')
TOLERANCE = 0.90


class BoardRecognition(unittest.TestCase):
    def test_all_boards(self):
        grid_files = sorted([f for f in os.listdir(GRID_DIR) if f.split('.')[1] == 'jpg'], key=lambda f: int(f.split('.')[0]))

        successes = 0
        total_time = 0
        missed_digits = 0
        for grid in grid_files:
            print('Processing %s...' % grid)

            start = time.time()
            sudoku = Sudoku(os.path.join(GRID_DIR, grid), DIGIT_MODEL)
            total_time += time.time() - start

            with open(os.path.join(GRID_DIR, '%s.dat' % grid.split('.')[0]), 'r') as f:
                board = solver.parse_sudoku_puzzle(f.read())

            # Add up the mis-classified digits
            diff = []
            guess = sudoku.board_dict
            for position in guess:
                if guess[position] != board[position]:
                    diff.append(position)
            missed_digits += len(diff)

            if sudoku.board_dict == board:
                successes += 1

        print('Success: %s\tFail: %s\tRatio: %s' % (successes, len(grid_files) - successes, successes / len(grid_files)))
        print('Elapsed: %ss\tAverage: %ss' % (total_time, total_time / len(grid_files)))
        print('Total Digits: %s\tMissed Digits: %s\tRatio: %s' % (len(grid_files) * 81, missed_digits, missed_digits / (len(grid_files) * 81)))

        self.assertGreaterEqual(successes / len(grid_files), TOLERANCE)


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(BoardRecognition)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
