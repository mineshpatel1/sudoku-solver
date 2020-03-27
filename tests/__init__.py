import unittest

import tests.SudokuSolver
import tests.DigitRecognition
import tests.BoardRecognition

if __name__ == '__main__':
    suites = [
        tests.SudokuSolver.suite(),
        tests.DigitRecognition.suite(),
        tests.BoardRecognition.suite()
    ]

    suite = unittest.TestSuite(suites)
    unittest.TextTestRunner().run(suite)
