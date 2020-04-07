import unittest
import os

import helper
import solver

PUZZLE_DIR = os.path.join('..', 'data', 'puzzles')


class SudokuSolver(unittest.TestCase):
    def test_grid(self):
        """Checks the grid references, units and peers are calculated properly."""
        squares, peers, units, all_units = solver.sudoku_elements()
        self.assertEqual(len(squares), 81)
        self.assertTrue(all(len(units[s]) == 3 for s in squares))
        self.assertTrue(all(len([cell for unit in units[s] for cell in unit]) == 27 for s in squares))
        self.assertTrue(all(len(peers[s]) == 20 for s in squares))

    def test_easy_puzzles(self):
        """Specific test cases used in development or that are otherwise uniquely interesting."""
        # Easy puzzle
        puzzle1 = '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'
        self.assertTrue(solver.validate_sudoku(solver.solve(puzzle1)))

        # Hard puzzle (requires brute-force search)
        puzzle2 = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'
        self.assertTrue(solver.validate_sudoku(solver.solve(puzzle2)))

        # Arto Inkala's 2006 Tough Sodokou
        puzzle3 = '85...24..72......9..4.........1.7..23.5...9...4...........8..7..17..........36.4.'
        self.assertTrue(solver.validate_sudoku(solver.solve(puzzle3)))

        # Arto Inkala's 2010 "Toughest Sudoku"
        puzzle4 = '..53.....8......2..7..1.5..4....53...1..7...6..32...8..6.5....9..4....3......97..'
        solved4 = solver.solve(puzzle4)
        self.assertTrue(solver.validate_sudoku(solved4))

        # Arto Inkala's 2012 "Even Tougher Sudoku"
        puzzle5 = '8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..'
        solved5 = solver.solve(puzzle5)
        self.assertTrue(solver.validate_sudoku(solved5))

        # Difficult for backtracking brute force algorithms
        puzzle6 = '..............3.85..1.2.......5.7.....4...1...9.......5......73..2.1........4...9'
        solved6 = solver.solve(puzzle6)
        self.assertTrue(solver.validate_sudoku(solved6))

    def test_non_unique(self):
        """Puzzles with more than 1 unique solution."""
        puzzle1 = '..4...6.....3..1.8.....2..5.........2...5..3...3.........93....1..8....9......7.2'
        self.assertTrue(solver.validate_sudoku(solver.solve(puzzle1)))

        puzzle2 = '............5.6.............18...34..............9......92.86..2..6.5..113.9.4.58'
        self.assertTrue(solver.validate_sudoku(solver.solve(puzzle2)))

        # Peter Norvig's toughest puzzle
        puzzle3 = '.....6....59.....82....8....45........3........6..3.54...325..6..................'
        self.assertTrue(solver.validate_sudoku(solver.solve(puzzle3)))

    def test_difficult(self):
        """Difficult puzzles for algorithms"""
        puzzles = []
        # Near worst case for brute-force solver (Wikipedia)
        puzzles.append('..............3.85..1.2.......5.7.....4...1...9.......5......73..2.1........4...9')

        # gsf's sudoku q1 (Platinum Blonde)
        puzzles.append('.......12........3..23..4....18....5.6..7.8.......9.....85.....9...4.5..47...6...')

        # (Cheese)
        puzzles.append('.2..5.7..4..1....68....3...2....8..3.4..2.5.....6...1...2.9.....9......57.4...9..')

        # (Fata Morgana)
        puzzles.append('........3..1..56...9..4..7......9.5.7.......8.5.4.2....8..2..9...35..1..6........')

        # (Red Dwarf)
        puzzles.append('12.3....435....1....4........54..2..6...7.........8.9...31..5.......9.7.....6...8')

        # (Easter Monster)
        puzzles.append('1.......2.9.4...5...6...7...5.9.3.......7.......85..4.7.....6...3...9.8...2.....1')

        # Nicolas Juillerat's Sudoku explainer 1.2.1 (top 5)
        puzzles.append('.......39.....1..5..3.5.8....8.9...6.7...2...1..4.......9.8..5..2....6..4..7.....')
        puzzles.append('12.3.....4.....3....3.5......42..5......8...9.6...5.7...15..2......9..6......7..8')
        puzzles.append('..3..6.8....1..2......7...4..9..8.6..3..4...1.7.2.....3....5.....5...6..98.....5.')
        puzzles.append('1.......9..67...2..8....4......75.3...5..2....6.3......9....8..6...4...1..25...6.')
        puzzles.append('..9...4...7.3...2.8...6...71..8....6....1..7.....56...3....5..1.4.....9...2...7..')

        # dukuso's suexrat9 (top 1)
        puzzles.append('....9..5..1.....3...23..7....45...7.8.....2.......64...9..1.....8..6......54....7')

        # From http://magictour.free.fr/topn87 (top 3)
        puzzles.append('4...3.......6..8..........1....5..9..8....6...7.2........1.27..5.3....4.9........')
        puzzles.append('7.8...3.....2.1...5.........4.....263...8.......1...9..9.6....4....7.5...........')
        puzzles.append('3.7.4...........918........4.....7.....16.......25..........38..9....5...2.6.....')

        # dukuso's suexratt (top 1)
        puzzles.append('........8..3...4...9..2..6.....79.......612...6.5.2.7...8...5...1.....2.4.5.....3')

        # First 2 from sudoku17
        puzzles.append('.......1.4.........2...........5.4.7..8...3....1.9....3..4..2...5.1........8.6...')
        puzzles.append('.......12....35......6...7.7.....3.....4..8..1...........12.....8.....4..5....6..')

        # 2 from http://www.setbb.com/phpbb/viewtopic.php?p=10478
        puzzles.append('1.......2.9.4...5...6...7...5.3.4.......6........58.4...2...6...3...9.8.7.......1')
        puzzles.append('.....1.2.3...4.5.....6....7..2.....1.8..9..3.4.....8..5....2....9..3.4....67.....')

        for puzzle in puzzles:
            self.assertTrue(solver.validate_sudoku(solver.solve(puzzle)))

    def test_benchmark(self):
        """Collection of all puzzles collected in this project."""
        total_time = 0
        n = 0
        for i, solution, elapsed in helper.benchmark_from_file(os.path.join(PUZZLE_DIR, 'benchmark.txt')):
            self.assertTrue(solver.validate_sudoku(solution))
            n = i
            total_time += elapsed

        print(f"Total Benchmark time: {total_time}s")
        print(f"Average Benchmark time: {1000 * (total_time / n)}ms")


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(SudokuSolver)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
